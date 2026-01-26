"""
Fine-tuning script for nano_hindi on Hindi Question Answering.

Uses IndicQA dataset with generative QA format.

Usage:
    python finetune.py --checkpoint checkpoints/checkpoint_step1678.pt --epochs 3
    python finetune.py --checkpoint checkpoints/checkpoint_step1678.pt --epochs 3 --wandb
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict
import re

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm

from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT
from finetune.config import FinetuneConfig
from finetune.dataset import create_dataloaders, QAExample


def load_pretrained_model(checkpoint_path: str, device: str = "cuda"):
    """Load pretrained nano_hindi model from checkpoint."""
    print(f"Loading pretrained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    config_dict = checkpoint["model_config"]
    config = GPTConfig(**{k: v for k, v in config_dict.items() if k != "head_dim"})

    print(f"Model config: {config}")

    # Create model
    model = GPT(config).to(device)

    # Handle torch.compile prefix (_orig_mod.) in state dict
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    print(f"Loaded from step {checkpoint['step']}, tokens seen: {checkpoint['tokens_seen']:,}")
    print(f"Model parameters: {model.num_params():,}")

    return model, config


def get_lr_schedule(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return step / warmup_steps
    if step >= total_steps:
        return min_lr_ratio

    import math
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    text = text.strip().lower()
    # Remove Hindi punctuation
    text = re.sub(r'[।॥,\.\?!:;\'\"\(\)\[\]{}]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_exact_match(pred: str, gold: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(pred) == normalize_answer(gold))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    val_examples: list,
    tokenizer,
    config: FinetuneConfig,
    device: str,
    max_samples: int = 100,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    f1_scores = []
    em_scores = []

    # Calculate loss on validation set
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids, labels)

        # Count valid tokens
        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)

    # Generate answers for a subset of examples
    samples_to_eval = min(max_samples, len(val_examples))

    for i in range(samples_to_eval):
        example = val_examples[i]

        # Format input
        input_text = (
            f"{config.context_prefix}{example.context}"
            f"{config.question_prefix}{example.question}"
            f"{config.answer_prefix}"
        )

        # Truncate if needed
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        if len(input_ids) > config.max_seq_len - config.eval_max_tokens:
            input_ids = input_ids[:config.max_seq_len - config.eval_max_tokens]

        # Generate
        generated = []
        for token in model.generate(
            input_ids,
            max_tokens=config.eval_max_tokens,
            temperature=config.eval_temperature,
            top_k=1,  # Greedy
        ):
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)

        # Decode prediction
        pred_text = tokenizer.decode(generated).strip()

        # Get gold answer (with filter_unanswerable=True, all examples have answers)
        gold_text = example.answer if example.answer else config.no_answer_text.strip()

        # Compute metrics
        f1_scores.append(compute_f1(pred_text, gold_text))
        em_scores.append(compute_exact_match(pred_text, gold_text))

    model.train()

    return {
        "loss": avg_loss,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    step: int,
    epoch: int,
    metrics: dict,
    config: FinetuneConfig,
    model_config: GPTConfig,
    is_best: bool = False,
):
    """Save fine-tuning checkpoint."""
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config.__dict__,
        "finetune_config": config.__dict__,
    }

    # Save latest
    checkpoint_path = save_dir / "latest.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Save best
    if is_best:
        best_path = save_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint: {best_path} (F1: {metrics['f1']:.4f})")


def finetune(args):
    """Main fine-tuning function."""
    device = args.device

    # Create config
    config = FinetuneConfig(
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_wandb=args.wandb,
    )

    print(f"Fine-tuning config: {config}")

    # Initialize wandb
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"finetune_qa_ep{config.epochs}",
            config={
                "finetune": config.__dict__,
            },
        )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Load pretrained model
    model, model_config = load_pretrained_model(config.checkpoint, device)
    model.train()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, val_examples = create_dataloaders(config, tokenizer)

    # Calculate total steps
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Setup optimizer (single AdamW, not dual Muon+AdamW)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
    )

    # Training loop
    print("\nStarting fine-tuning...")
    step = 0
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")

        epoch_loss = 0.0
        micro_step = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids, labels)
                loss = loss / config.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            epoch_loss += loss.item()
            micro_step += 1

            # Gradient accumulation
            if micro_step % config.gradient_accumulation_steps == 0:
                # Update learning rate
                lr_mult = get_lr_schedule(step, warmup_steps, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr * lr_mult

                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                step += 1

                # Logging
                if step % config.log_interval == 0:
                    avg_loss = epoch_loss / micro_step * config.gradient_accumulation_steps
                    current_lr = optimizer.param_groups[0]["lr"]

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                    })

                    if config.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": current_lr,
                            "train/step": step,
                            "train/epoch": epoch + 1,
                        })

                # Evaluation
                if step % config.eval_interval == 0:
                    metrics = evaluate(
                        model, val_loader, val_examples, tokenizer, config, device
                    )

                    print(f"\n  Step {step}: Val loss={metrics['loss']:.4f}, F1={metrics['f1']:.4f}, EM={metrics['em']:.4f}")

                    # Save best checkpoint
                    is_best = metrics["f1"] > best_f1
                    if is_best:
                        best_f1 = metrics["f1"]

                    save_checkpoint(
                        model, optimizer, step, epoch,
                        metrics, config, model_config, is_best
                    )

                    if config.use_wandb:
                        import wandb
                        wandb.log({
                            "val/loss": metrics["loss"],
                            "val/f1": metrics["f1"],
                            "val/em": metrics["em"],
                            "train/step": step,
                        })

        # End of epoch evaluation
        print(f"\nEnd of epoch {epoch + 1}")
        metrics = evaluate(
            model, val_loader, val_examples, tokenizer, config, device
        )
        print(f"  Val loss={metrics['loss']:.4f}, F1={metrics['f1']:.4f}, EM={metrics['em']:.4f}")

        is_best = metrics["f1"] > best_f1
        if is_best:
            best_f1 = metrics["f1"]

        save_checkpoint(
            model, optimizer, step, epoch,
            metrics, config, model_config, is_best
        )

    # Final summary
    elapsed = time.time() - start_time
    print(f"\nFine-tuning complete!")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Best checkpoint: {config.save_dir}/best.pt")

    if config.use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune nano_hindi for QA")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_step1678.pt",
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
