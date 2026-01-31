"""
Common utilities for distributed training and logging.
"""

import os
import torch
import torch.distributed as dist


def get_dist_info():
    """Get distributed training info."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    return ddp, rank, local_rank, world_size


def print0(*args, **kwargs):
    """Print only on rank 0."""
    ddp, rank, _, _ = get_dist_info()
    if not ddp or rank == 0:
        print(*args, **kwargs)


def setup_distributed():
    """Initialize distributed training if available."""
    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Performance settings for H100 / modern GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    return ddp, rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
