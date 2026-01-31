"""
Configuration classes for nano_hindi model and training.

Model sizes (with 68K Sarvam vocab):
- 22m: 6 layers, 256 dim, tied embeddings, no value embeds
- 36m: 6 layers, 384 dim, tied embeddings, no value embeds
- 45m: 8 layers, 384 dim, tied embeddings, no value embeds
- 250m: 32 layers, 768 dim, tied embeddings, no value embeds

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
    n_layer: int = 6
    n_head: int = 4  # Query heads
    n_kv_head: int = 2  # Key/Value heads for GQA
    n_embd: int = 256

    # Sequence length
    sequence_len: int = 1024

    # Sliding window attention pattern
    window_pattern: str = "SSSL"

    # Architecture options (to control param count)
    tie_embeddings: bool = True  # Share wte and lm_head weights
    use_value_embeds: bool = False  # ResFormer-style value embeddings

    # Derived (computed in __post_init__)
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    @property
    def n_params(self) -> int:
        """Estimate total parameters."""
        kv_dim = self.n_kv_head * self.head_dim

        # Embeddings
        embed_params = self.vocab_size * self.n_embd  # Token embedding (wte)
        if not self.tie_embeddings:
            embed_params += self.vocab_size * self.n_embd  # LM head (untied)

        # Value embeddings (alternating layers, ~half)
        if self.use_value_embeds:
            n_ve_layers = (self.n_layer + 1) // 2
            embed_params += n_ve_layers * self.vocab_size * kv_dim

        # Per-layer params
        attn_params = self.n_embd * (self.n_head * self.head_dim)  # Q
        attn_params += self.n_embd * (self.n_kv_head * self.head_dim)  # K
        attn_params += self.n_embd * (self.n_kv_head * self.head_dim)  # V
        attn_params += self.n_embd * self.n_embd  # Output

        # MLP: up + down projections (4x expansion)
        mlp_params = self.n_embd * (4 * self.n_embd) * 2

        # Per-layer scalars + VE gate
        scalar_params = 2
        if self.use_value_embeds:
            scalar_params += 32 * self.n_kv_head  # VE gate

        layer_params = attn_params + mlp_params + scalar_params

        return embed_params + self.n_layer * layer_params

    def __repr__(self):
        return (
            f"GPTConfig(~{self.n_params / 1e6:.1f}M params, "
            f"{self.n_layer}L, {self.n_embd}d, {self.n_head}/{self.n_kv_head}h, "
            f"tied={self.tie_embeddings}, VE={self.use_value_embeds})"
        )


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_dir: str = "data"
    train_file: str = "train.bin"
    val_file: str = "val.bin"

    # Batch size (reduced to fit in 24GB GPU with 68K vocab)
    batch_size: int = 16
    gradient_accumulation_steps: int = 16  # Effective batch = 16 * 16 = 256

    # Training duration (Chinchilla 20:1 scaling)
    total_tokens: int = 440_000_000  # ~440M for 22M model

    # Learning rates
    muon_lr: float = 0.02
    adamw_embedding_lr: float = 0.2
    adamw_unembedding_lr: float = 0.004
    adamw_scalar_lr: float = 0.5

    # Optimizer settings
    weight_decay: float = 0.0
    adam_betas: tuple = (0.8, 0.95)
    muon_momentum: float = 0.95

    # Learning rate schedule
    warmup_steps: int = 100  # ~5% of training
    min_lr_ratio: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    keep_last_n_checkpoints: int = 3

    # Evaluation and inference during training
    eval_interval: int = 100
    inference_interval: int = 200
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
    dtype: str = "bfloat16"
    compile_model: bool = True

    # Reproducibility
    seed: int = 42

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps * 1024

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.tokens_per_step


# Preset configurations
CONFIGS = {
    # ~22M params - smallest practical with 68K vocab
    "22m": GPTConfig(
        n_layer=6,
        n_head=4,
        n_kv_head=2,
        n_embd=256,
        tie_embeddings=True,
        use_value_embeds=False,
    ),
    # ~36M params - better quality
    "36m": GPTConfig(
        n_layer=6,
        n_head=6,
        n_kv_head=2,
        n_embd=384,
        tie_embeddings=True,
        use_value_embeds=False,
    ),
    # ~45M params - deeper variant
    "45m": GPTConfig(
        n_layer=8,
        n_head=6,
        n_kv_head=2,
        n_embd=384,
        tie_embeddings=True,
        use_value_embeds=False,
    ),
    # ~250M params - scaled up for 8×H100 training
    "250m": GPTConfig(
        n_layer=32,
        n_head=12,
        n_kv_head=4,
        n_embd=768,
        tie_embeddings=True,
        use_value_embeds=False,
    ),
}


def get_config(size: str = "22m") -> GPTConfig:
    """Get a preset model configuration."""
    if size not in CONFIGS:
        raise ValueError(f"Unknown config: {size}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[size]
