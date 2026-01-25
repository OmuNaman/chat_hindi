"""
Training script for nano_hindi.

Features:
- Dual optimizer setup (Muon + AdamW)
- Checkpointing with configurable interval
- Inference samples during training
- BPB evaluation
- Wandb logging
- torch.compile for speedup

Usage:
    python train.py --config 25m --total_tokens 500000000
"""

import argparse
import os
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm

from nano_hindi.config import GPTConfig, TrainConfig, get_config
from nano_hindi.model import GPT
from nano_hindi.muon import Muon, DistMuon
from nano_hindi.eval import evaluate_loss, compute_token_bytes
from nano_hindi.common import setup_distributed, cleanup_distributed, print0


class DataLoader:
    """Memory-mapped data loader for efficient training."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        dtype=np.uint32,
    ):
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.n_tokens = len(self.data)

    def get_batch(self):
        """Get a random batch of sequences."""
        ix = torch.randint(self.n_tokens - self.seq_len - 1, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(self.data[i : i + self.seq_len].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64)
                )
                for i in ix
            ]
        )
        return x.to(self.device), y.to(self.device)

    def __iter__(self):
        while True:
            yield self.get_batch()


def get_lr_schedule(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return step / warmup_steps
    if step >= total_steps:
        return min_lr_ratio

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress))


def setup_optimizers(model: GPT, config: TrainConfig, ddp: bool = False):
    """Setup Muon optimizer for weights and AdamW for embeddings."""
    model_dim = model.config.n_embd

    # Separate parameters
    matrix_params = list(model.transformer.h.parameters())
    value_embeds_params = list(model.value_embeds.parameters()) if model.value_embeds else []
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters()) if model.lm_head is not None else []
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]

    # Scale LR by 1/sqrt(d_model/768)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print0(f"LR scale for AdamW: {dmodel_lr_scale:.4f}")

    # AdamW for embeddings and scalars
    adam_groups = [
        dict(params=embedding_params, lr=config.adamw_embedding_lr * dmodel_lr_scale),
        dict(params=resid_params, lr=config.adamw_scalar_lr * 0.01),
        dict(params=x0_params, lr=config.adamw_scalar_lr, betas=(0.96, 0.95)),
    ]
    # Only add lm_head params if not tied
    if lm_head_params:
        adam_groups.insert(0, dict(params=lm_head_params, lr=config.adamw_unembedding_lr * dmodel_lr_scale))
    # Only add value_embeds params if they exist
    if value_embeds_params:
        adam_groups.append(dict(params=value_embeds_params, lr=config.adamw_embedding_lr * dmodel_lr_scale))
    adamw_optimizer = torch.optim.AdamW(
        adam_groups,
        betas=config.adam_betas,
        eps=1e-10,
        weight_decay=0.0,
        fused=True,
    )

    # Muon for attention/MLP weights
    MuonClass = DistMuon if ddp else Muon
    muon_optimizer = MuonClass(
        matrix_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum,
        weight_decay=config.weight_decay,
    )

    # Store initial LRs for scheduling
    for opt in [adamw_optimizer, muon_optimizer]:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return adamw_optimizer, muon_optimizer


def save_checkpoint(
    model: nn.Module,
    adamw_opt,
    muon_opt,
    step: int,
    tokens_seen: int,
    loss: float,
    config: TrainConfig,
    model_config: GPTConfig,
):
    """Save training checkpoint."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "adamw_optimizer": adamw_opt.state_dict(),
        "muon_optimizer": muon_opt.state_dict(),
        "model_config": model_config.__dict__,
        "train_config": config.__dict__,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print0(f"Saved checkpoint: {checkpoint_path}")

    # Clean up old checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    if len(checkpoints) > config.keep_last_n_checkpoints:
        for old_ckpt in checkpoints[: -config.keep_last_n_checkpoints]:
            old_ckpt.unlink()
            print0(f"Removed old checkpoint: {old_ckpt}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, adamw_opt, muon_opt, device: str):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    adamw_opt.load_state_dict(checkpoint["adamw_optimizer"])
    muon_opt.load_state_dict(checkpoint["muon_optimizer"])

    return checkpoint["step"], checkpoint["tokens_seen"]


def run_inference(model: nn.Module, tokenizer, prompts: list, max_tokens: int = 50, temperature: float = 0.8):
    """Generate text samples for monitoring during training."""
    model_to_use = model.module if hasattr(model, "module") else model
    model_to_use.eval()

    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated = []

        for token in model_to_use.generate(
            input_ids, max_tokens=max_tokens, temperature=temperature, top_k=50
        ):
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break

        output = tokenizer.decode(input_ids + generated)
        results.append((prompt, output))

    model_to_use.train()
    return results


def train(args):
    """Main training function."""
    # Setup distributed if available
    ddp, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    # Device setup
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device) if torch.cuda.is_available() else None

    # Load configs
    model_config = get_config(args.config)
    train_config = TrainConfig(
        total_tokens=args.total_tokens,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb and is_main,
    )

    # Override batch_size if provided
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size

    # Override compile if --no_compile
    if args.no_compile:
        train_config.compile_model = False

    print0(f"Model config: {model_config}")
    print0(f"Training for {train_config.total_tokens:,} tokens")
    print0(f"Batch size: {train_config.batch_size} x {train_config.gradient_accumulation_steps} = {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print0(f"Total steps: {train_config.total_steps:,}")

    # Initialize wandb
    if train_config.use_wandb:
        import wandb
        wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name or f"nano_hindi_{args.config}",
            config={
                "model": model_config.__dict__,
                "training": train_config.__dict__,
            },
        )

    # Load tokenizer
    print0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

    # Create model
    print0("Creating model...")
    model = GPT(model_config).to(device)
    model.init_weights()
    print0(f"Model parameters: {model.num_params():,}")

    # Compile model
    if train_config.compile_model and torch.cuda.is_available():
        print0("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap in DDP if distributed
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Setup optimizers
    adamw_opt, muon_opt = setup_optimizers(
        model.module if ddp else model,
        train_config,
        ddp=ddp,
    )

    # Setup data loaders
    print0("Loading data...")
    train_loader = DataLoader(
        Path(train_config.data_dir) / train_config.train_file,
        batch_size=train_config.batch_size,
        seq_len=model_config.sequence_len,
        device=device,
    )
    val_loader = DataLoader(
        Path(train_config.data_dir) / train_config.val_file,
        batch_size=train_config.batch_size,
        seq_len=model_config.sequence_len,
        device=device,
    )

    # Resume from checkpoint if exists
    start_step = 0
    tokens_seen = 0
    latest_ckpt = None
    ckpt_dir = Path(train_config.checkpoint_dir)
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint_step*.pt"))
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            print0(f"Resuming from {latest_ckpt}")
            start_step, tokens_seen = load_checkpoint(
                str(latest_ckpt), model, adamw_opt, muon_opt, device
            )

    # Training loop
    print0("Starting training...")
    model.train()

    micro_step = 0
    running_loss = 0.0
    start_time = time.time()

    train_iter = iter(train_loader)

    for step in range(start_step, train_config.total_steps):
        # Update learning rate
        lr_mult = get_lr_schedule(
            step,
            train_config.warmup_steps,
            train_config.total_steps,
            train_config.min_lr_ratio,
        )
        for opt in [adamw_opt, muon_opt]:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lr_mult

        # Gradient accumulation
        for micro in range(train_config.gradient_accumulation_steps):
            x, y = next(train_iter)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
                loss = loss / train_config.gradient_accumulation_steps

            loss.backward()
            running_loss += loss.item()

            micro_step += 1
            tokens_seen += x.numel()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        adamw_opt.step()
        muon_opt.step()
        adamw_opt.zero_grad()
        muon_opt.zero_grad()

        # Logging
        if step % train_config.log_interval == 0 and is_main:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_seen / elapsed
            current_lr = adamw_opt.param_groups[0]["lr"]

            # Average loss over log_interval steps (fix: was accumulating instead of averaging)
            avg_loss = running_loss / max(1, train_config.log_interval if step > 0 else 1)

            log_msg = (
                f"Step {step:>6d} | "
                f"Loss {avg_loss:.4f} | "
                f"LR {current_lr:.2e} | "
                f"Tokens {tokens_seen/1e6:.1f}M | "
                f"Speed {tokens_per_sec/1e3:.1f}K tok/s"
            )
            print(log_msg)

            if train_config.use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/tokens": tokens_seen,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step,
                })

            running_loss = 0.0

        # Evaluation
        if step % train_config.eval_interval == 0 and step > 0 and is_main:
            val_loss = evaluate_loss(
                model.module if ddp else model,
                val_loader,
                steps=20,
            )
            print(f"  Val loss: {val_loss:.4f}")

            if train_config.use_wandb:
                wandb.log({"val/loss": val_loss, "train/step": step})

        # Inference samples
        if step % train_config.inference_interval == 0 and step > 0 and is_main:
            print("\n--- Inference Samples ---")
            samples = run_inference(
                model,
                tokenizer,
                train_config.inference_prompts,
                max_tokens=train_config.inference_max_tokens,
            )
            for prompt, output in samples:
                print(f"Prompt: {prompt}")
                print(f"Output: {output}")
                print()

            if train_config.use_wandb:
                table = wandb.Table(columns=["prompt", "output"])
                for prompt, output in samples:
                    table.add_data(prompt, output)
                wandb.log({"inference/samples": table, "train/step": step})

            print("-------------------------\n")

        # Checkpointing
        if step % train_config.checkpoint_interval == 0 and step > 0 and is_main:
            save_checkpoint(
                model,
                adamw_opt,
                muon_opt,
                step,
                tokens_seen,
                running_loss,
                train_config,
                model_config,
            )

    # Final checkpoint
    if is_main:
        save_checkpoint(
            model,
            adamw_opt,
            muon_opt,
            train_config.total_steps,
            tokens_seen,
            running_loss,
            train_config,
            model_config,
        )

    print0("Training complete!")

    # Cleanup
    if train_config.use_wandb:
        wandb.finish()
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train nano_hindi model")
    parser.add_argument(
        "--config",
        type=str,
        default="22m",
        choices=["22m", "36m", "45m"],
        help="Model configuration (22m=~22M params, 36m=~36M, 45m=~45M)",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=440_000_000,
        help="Total tokens to train on (default: 440M for 22M model)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (default: from config)",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (saves memory)",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
