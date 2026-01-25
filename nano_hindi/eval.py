"""
Evaluation utilities for nano_hindi.

Main metric: Bits Per Byte (BPB) - tokenization-independent evaluation.
"""

import math
import torch
import torch.distributed as dist


@torch.no_grad()
def evaluate_bpb(model, batches, steps: int, token_bytes: torch.Tensor) -> float:
    """
    Calculate Bits Per Byte (BPB) - a tokenization-independent metric.

    BPB = total_nats / (ln(2) * total_bytes)

    Args:
        model: The language model
        batches: Iterator yielding (x, y) batches
        steps: Number of evaluation steps
        token_bytes: 1D tensor of shape (vocab_size,) with byte count per token
                    (0 for special tokens that shouldn't be counted)

    Returns:
        BPB value (lower is better)
    """
    model.eval()
    device = model.get_device()

    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)

        # Get per-token loss
        loss2d = model(x, y, loss_reduction="none").view(-1)
        y = y.view(-1)

        # Handle ignored tokens (target < 0)
        if (y.int() < 0).any():
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype),
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()

    # All-reduce across ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    # Calculate BPB
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    if total_bytes == 0:
        return float("inf")

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


def compute_token_bytes(tokenizer) -> torch.Tensor:
    """
    Compute byte length for each token in the vocabulary.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        1D tensor of shape (vocab_size,) with byte counts
        Special tokens have value 0 (excluded from BPB calculation)
    """
    vocab_size = tokenizer.vocab_size
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32)

    special_token_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_token_ids = set(tokenizer.all_special_ids)

    for token_id in range(vocab_size):
        if token_id in special_token_ids:
            token_bytes[token_id] = 0
            continue

        try:
            text = tokenizer.decode([token_id])
            token_bytes[token_id] = len(text.encode("utf-8"))
        except Exception:
            token_bytes[token_id] = 0

    return token_bytes


@torch.no_grad()
def evaluate_loss(model, batches, steps: int) -> float:
    """
    Simple average cross-entropy loss evaluation.

    Args:
        model: The language model
        batches: Iterator yielding (x, y) batches
        steps: Number of evaluation steps

    Returns:
        Average loss
    """
    model.eval()
    device = model.get_device()

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss = model(x, y, loss_reduction="sum")
        total_loss += loss
        total_tokens += (y != -1).sum()

    # All-reduce
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    return (total_loss / total_tokens).item()
