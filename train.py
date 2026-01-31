"""
Training script for nano_hindi.

Optimized for 2×H100 SXM multi-GPU training with:
- Async data prefetching with pinned memory
- DDP no_sync() to avoid redundant gradient syncs
- torch.compile with max-autotune mode
- TF32 matmul + cuDNN benchmark for H100
- Non-blocking CPU→GPU transfers
- Cached parameter lists for grad clipping

Usage:
    # Single GPU
    python train.py --config 250m --total_tokens 5000000000 --wandb

    # 2×H100 with DDP
    torchrun --nproc_per_node=2 train.py --config 250m --total_tokens 5000000000 --batch_size 64 --wandb
"""

import argparse
import os
import time
import threading
from contextlib import nullcontext
from pathlib import Path
from queue import Queue

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from nano_hindi.config import GPTConfig, TrainConfig, get_config
from nano_hindi.model import GPT
from nano_hindi.muon import Muon, DistMuon
from nano_hindi.eval import evaluate_loss, compute_token_bytes
from nano_hindi.common import setup_distributed, cleanup_distributed, print0


class DataLoader:
    """Memory-mapped data loader with async prefetching and pinned memory."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        seq_len: int,
        device: str = "cuda",
        dtype=np.uint32,
        prefetch_batches: int = 2,
    ):
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.n_tokens = len(self.data)
        self.prefetch_batches = prefetch_batches

        # Prefetch queue holds independent tensors (no buffer reuse race)
        self._queue = Queue(maxsize=prefetch_batches)
        self._stop = threading.Event()
        self._thread = None

    def _make_batch(self):
        """Create a batch in pinned memory using vectorized numpy ops."""
        ix = np.random.randint(0, self.n_tokens - self.seq_len - 1, size=self.batch_size)
        # Vectorized: gather all sequences at once
        offsets = np.arange(self.seq_len + 1)  # [0, 1, ..., seq_len]
        indices = ix[:, None] + offsets[None, :]  # (batch, seq_len+1)
        chunks = self.data[indices].astype(np.int64)  # (batch, seq_len+1)
        x = torch.from_numpy(chunks[:, :-1].copy()).pin_memory()
        y = torch.from_numpy(chunks[:, 1:].copy()).pin_memory()
        return x, y

    def _prefetch_loop(self):
        """Background thread that prefetches batches into fresh pinned tensors."""
        while not self._stop.is_set():
            x, y = self._make_batch()
            try:
                self._queue.put((x, y), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    def start_prefetch(self):
        """Start the background prefetch thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def stop_prefetch(self):
        """Stop the background prefetch thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def get_batch(self):
        """Get a batch, using prefetched data if available."""
        if self._thread is not None and self._thread.is_alive():
            try:
                x, y = self._queue.get(timeout=10.0)
            except Exception:
                # Prefetch thread may have died; fallback to sync
                x, y = self._make_batch()
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        else:
            x, y = self._make_batch()
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def __iter__(self):
        while True:
            yield self.get_batch()


def get_lr_schedule(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """Cosine learning rate schedule with warmup."""
    if warmup_steps > 0 and step < warmup_steps:
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


def _state_dict_to_cpu(state_dict):
    """Recursively move all tensors in a state dict to CPU."""
    result = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            result[k] = v.detach().cpu()
        elif isinstance(v, dict):
            result[k] = _state_dict_to_cpu(v)
        elif isinstance(v, list):
            result[k] = [t.detach().cpu() if torch.is_tensor(t) else t for t in v]
        else:
            result[k] = v
    return result


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
    """Save training checkpoint (non-blocking via background thread)."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Move ALL state to CPU before handing off to background thread
    # (touching CUDA tensors from a non-main thread can cause random crashes)
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    checkpoint = {
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "model_state_dict": {k: v.cpu() for k, v in raw_model.state_dict().items()},
        "adamw_optimizer": _state_dict_to_cpu(adamw_opt.state_dict()),
        "muon_optimizer": _state_dict_to_cpu(muon_opt.state_dict()),
        "model_config": model_config.__dict__,
        "train_config": config.__dict__,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_step{step}.pt"

    def _save():
        # Atomic write: save to temp file, then rename
        tmp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(checkpoint_path)
        # Clean up old checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
        if len(checkpoints) > config.keep_last_n_checkpoints:
            for old_ckpt in checkpoints[: -config.keep_last_n_checkpoints]:
                old_ckpt.unlink()

    thread = threading.Thread(target=_save, daemon=True)
    thread.start()
    print0(f"Saving checkpoint: {checkpoint_path} (async)")
    return thread


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
    if hasattr(model_to_use, "_orig_mod"):
        model_to_use = model_to_use._orig_mod
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
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Per-rank seeding so each GPU samples different data
    seed = 1337 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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

    # Compute total_steps accounting for world_size
    # (TrainConfig.total_steps ignores world_size, so we compute it here)
    tokens_per_step = (
        train_config.batch_size
        * train_config.gradient_accumulation_steps
        * world_size
        * model_config.sequence_len
    )
    total_steps = train_config.total_tokens // tokens_per_step

    print0(f"Model config: {model_config}")
    print0(f"Training for {train_config.total_tokens:,} tokens")
    print0(f"Batch size: {train_config.batch_size} x {train_config.gradient_accumulation_steps} x {world_size} GPUs = {train_config.batch_size * train_config.gradient_accumulation_steps * world_size} sequences/step")
    print0(f"Tokens per step: {tokens_per_step:,}")
    print0(f"Total steps: {total_steps:,}")

    # Initialize wandb
    if train_config.use_wandb:
        import wandb
        wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name or f"nano_hindi_{args.config}",
            config={
                "model": model_config.__dict__,
                "training": train_config.__dict__,
                "world_size": world_size,
            },
        )

    # Load tokenizer (only rank 0 needs it for inference samples)
    tokenizer = None
    if is_main:
        print0("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

    # Create model
    print0("Creating model...")
    model = GPT(model_config).to(device)
    model.init_weights()
    print0(f"Model parameters: {model.num_params():,}")

    # Setup optimizers on raw model BEFORE compile/DDP wrapping
    adamw_opt, muon_opt = setup_optimizers(model, train_config, ddp=ddp)

    # Compile FIRST, then wrap in DDP
    # (compile after DDP breaks no_sync() and state_dict on PyTorch 2.4)
    if train_config.compile_model and torch.cuda.is_available():
        print0("Compiling model with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    if ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    # Cache all parameters for fast grad clipping (avoid re-iterating every step)
    all_params = [p for p in model.parameters() if p.requires_grad]
    print0(f"Trainable parameter tensors: {len(all_params)}")

    # Setup data loaders with async prefetching
    print0("Loading data...")
    train_loader = DataLoader(
        Path(train_config.data_dir) / train_config.train_file,
        batch_size=train_config.batch_size,
        seq_len=model_config.sequence_len,
        device=device,
        prefetch_batches=3,
    )
    val_loader = DataLoader(
        Path(train_config.data_dir) / train_config.val_file,
        batch_size=train_config.batch_size,
        seq_len=model_config.sequence_len,
        device=device,
    )

    # Start async prefetch for training data
    train_loader.start_prefetch()
    print0("Data prefetch started (async)")

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

    # Pre-compute no_sync context for DDP gradient accumulation
    # Only sync gradients on the LAST micro-step (saves N-1 all-reduces per step)
    no_sync_ctx = model.no_sync if ddp else nullcontext

    # Training loop
    print0("Starting training...")
    model.train()

    micro_step = 0
    running_loss = 0.0
    last_avg_loss = float("inf")
    start_time = time.time()
    ckpt_thread = None  # Track async checkpoint thread

    train_iter = iter(train_loader)

    for step in range(start_step, total_steps):
        # Update learning rate
        lr_mult = get_lr_schedule(
            step,
            train_config.warmup_steps,
            total_steps,
            train_config.min_lr_ratio,
        )
        for opt in [adamw_opt, muon_opt]:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lr_mult

        # Gradient accumulation with DDP no_sync optimization
        accumulated_loss = torch.zeros(1, device=device)
        for micro in range(train_config.gradient_accumulation_steps):
            x, y = next(train_iter)

            # Skip gradient sync for all micro-steps except the last one
            is_last_micro = (micro == train_config.gradient_accumulation_steps - 1)
            sync_ctx = nullcontext() if is_last_micro else no_sync_ctx()

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                    loss = loss / train_config.gradient_accumulation_steps

                loss.backward()

            accumulated_loss += loss.detach()  # Stay on GPU, no sync
            micro_step += 1
            tokens_seen += x.numel() * world_size  # Count across all GPUs

        # Single GPU→CPU sync per optimizer step instead of per micro-step
        running_loss += accumulated_loss.item()

        # Gradient clipping (using cached param list)
        nn.utils.clip_grad_norm_(all_params, 1.0)

        # Optimizer step
        adamw_opt.step()
        muon_opt.step()
        adamw_opt.zero_grad(set_to_none=True)  # set_to_none=True saves memory
        muon_opt.zero_grad(set_to_none=True)

        # Logging
        if step % train_config.log_interval == 0 and is_main:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_seen / elapsed
            current_lr = adamw_opt.param_groups[0]["lr"]

            avg_loss = running_loss / max(1, train_config.log_interval if step > 0 else 1)

            log_msg = (
                f"Step {step:>6d}/{total_steps} | "
                f"Loss {avg_loss:.4f} | "
                f"LR {current_lr:.2e} | "
                f"Tokens {tokens_seen/1e9:.3f}B | "
                f"Speed {tokens_per_sec/1e3:.1f}K tok/s | "
                f"GPU mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            )
            print(log_msg)

            if train_config.use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/tokens": tokens_seen,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step,
                    "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                })

            last_avg_loss = avg_loss
            running_loss = 0.0

        # Evaluation
        if step % train_config.eval_interval == 0 and step > 0 and is_main:
            eval_model = model.module if ddp else model
            if hasattr(eval_model, "_orig_mod"):
                eval_model = eval_model._orig_mod
            val_loss = evaluate_loss(eval_model, val_loader, steps=20)
            model.train()  # Restore train mode (evaluate_loss sets eval mode)
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

        # Checkpointing (async save)
        if step % train_config.checkpoint_interval == 0 and step > 0 and is_main:
            # Wait for previous checkpoint to finish if still writing
            if ckpt_thread is not None:
                ckpt_thread.join()
            ckpt_thread = save_checkpoint(
                model,
                adamw_opt,
                muon_opt,
                step,
                tokens_seen,
                last_avg_loss,
                train_config,
                model_config,
            )

    # Stop prefetch thread
    train_loader.stop_prefetch()

    # Final checkpoint
    if is_main:
        if ckpt_thread is not None:
            ckpt_thread.join()
        ckpt_thread = save_checkpoint(
            model,
            adamw_opt,
            muon_opt,
            total_steps,
            tokens_seen,
            last_avg_loss,
            train_config,
            model_config,
        )
        ckpt_thread.join()  # Wait for final checkpoint

    print0("Training complete!")
    print0(f"Total tokens: {tokens_seen:,}")
    print0(f"Total time: {time.time() - start_time:.1f}s")
    print0(f"Avg throughput: {tokens_seen / (time.time() - start_time) / 1e3:.1f}K tok/s")

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
        choices=["22m", "36m", "45m", "250m"],
        help="Model configuration (22m=~22M, 36m=~36M, 45m=~45M, 250m=~250M)",
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
