"""
Configuration classes for nano_hindi model and training.

Easy to modify for different model sizes:
- 25M: n_layer=8, n_embd=384 (default, for testing)
- 40M: n_layer=10, n_embd=512
- 50M: n_layer=12, n_embd=576

Chinchilla scaling: tokens = 20 * params
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPTConfig:
    """Model architecture configuration."""

    # Tokenizer (Sarvam-1)
    vocab_size: int = 68096

    # Model dimensions
    n_layer: int = 8
    n_head: int = 6  # Query heads
    n_kv_head: int = 2  # Key/Value heads for GQA (3:1 ratio)
    n_embd: int = 384

    # Sequence length
    sequence_len: int = 1024

    # Sliding window attention pattern
    # Characters: L=long (full context), S=short (half context)
    # Pattern is tiled across layers, final layer always gets L
    window_pattern: str = "SSSL"

    # Derived (computed in __post_init__)
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    @property
    def n_params(self) -> int:
        """Estimate total parameters."""
        # Embeddings (token + value embeddings for ~half the layers)
        n_ve_layers = (self.n_layer + 1) // 2  # Alternating layers
        kv_dim = self.n_kv_head * self.head_dim
        embed_params = self.vocab_size * self.n_embd  # Token embedding
        embed_params += self.vocab_size * self.n_embd  # LM head (untied)
        embed_params += n_ve_layers * self.vocab_size * kv_dim  # Value embeddings

        # Per-layer params
        # Attention: Q, K, V projections + output projection
        attn_params = self.n_embd * (self.n_head * self.head_dim)  # Q
        attn_params += self.n_embd * (self.n_kv_head * self.head_dim)  # K
        attn_params += self.n_embd * (self.n_kv_head * self.head_dim)  # V
        attn_params += self.n_embd * self.n_embd  # Output

        # MLP: up + down projections (4x expansion)
        mlp_params = self.n_embd * (4 * self.n_embd)  # Up
        mlp_params += (4 * self.n_embd) * self.n_embd  # Down

        # Per-layer scalars
        scalar_params = 2  # resid_lambda + x0_lambda

        layer_params = attn_params + mlp_params + scalar_params

        return embed_params + self.n_layer * layer_params

    def __repr__(self):
        return (
            f"GPTConfig(~{self.n_params / 1e6:.1f}M params, "
            f"{self.n_layer} layers, {self.n_embd} dim, "
            f"{self.n_head}/{self.n_kv_head} heads)"
        )


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_dir: str = "data"
    train_file: str = "train.bin"
    val_file: str = "val.bin"

    # Batch size
    batch_size: int = 64  # Micro batch size
    gradient_accumulation_steps: int = 4  # Effective batch = 64 * 4 = 256

    # Training duration (Chinchilla 20:1 scaling)
    total_tokens: int = 500_000_000  # 500M tokens for ~25M model

    # Learning rates (from modded-nanogpt)
    muon_lr: float = 0.02  # For attention/MLP weights
    adamw_embedding_lr: float = 0.2  # For token embeddings
    adamw_unembedding_lr: float = 0.004  # For lm_head
    adamw_scalar_lr: float = 0.5  # For per-layer scalars

    # Optimizer settings
    weight_decay: float = 0.0
    adam_betas: tuple = (0.8, 0.95)
    muon_momentum: float = 0.95

    # Learning rate schedule
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # Min LR = 0.1 * max LR

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000  # Save every N steps
    keep_last_n_checkpoints: int = 3

    # Evaluation and inference during training
    eval_interval: int = 500  # Evaluate every N steps
    inference_interval: int = 1000  # Generate samples every N steps
    inference_prompts: list = field(default_factory=lambda: [
        "भारत एक",
        "आज का मौसम",
        "हिंदी भाषा",
    ])
    inference_max_tokens: int = 50

    # Logging
    log_interval: int = 10
    use_wandb: bool = True
    wandb_project: str = "nano_hindi"
    wandb_run_name: Optional[str] = None

    # Device and precision
    device: str = "cuda"
    dtype: str = "bfloat16"  # bfloat16 or float16
    compile_model: bool = True  # torch.compile for speedup

    # Reproducibility
    seed: int = 42

    @property
    def tokens_per_step(self) -> int:
        """Tokens processed per optimizer step."""
        return self.batch_size * self.gradient_accumulation_steps * 1024  # seq_len

    @property
    def total_steps(self) -> int:
        """Total optimizer steps."""
        return self.total_tokens // self.tokens_per_step


# Preset configurations for different model sizes
CONFIGS = {
    "25m": GPTConfig(
        n_layer=8,
        n_head=6,
        n_kv_head=2,
        n_embd=384,
    ),
    "40m": GPTConfig(
        n_layer=10,
        n_head=8,
        n_kv_head=2,
        n_embd=512,
    ),
    "50m": GPTConfig(
        n_layer=12,
        n_head=8,
        n_kv_head=2,
        n_embd=576,
    ),
}


def get_config(size: str = "25m") -> GPTConfig:
    """Get a preset model configuration."""
    if size not in CONFIGS:
        raise ValueError(f"Unknown config size: {size}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[size]
