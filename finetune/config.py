"""Fine-tuning configuration for nano_hindi QA."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning nano_hindi on QA tasks."""

    # Base model
    checkpoint: str = "checkpoints/checkpoint_step1678.pt"
    tokenizer_name: str = "sarvamai/sarvam-1"

    # Data
    data_path: str = "data/qa/indicqa.hi.json"
    train_split: float = 0.9  # 90% train, 10% val
    max_seq_len: int = 512  # Max sequence length for QA
    filter_unanswerable: bool = True  # Remove questions with no answer (category="NO")

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch = 32
    epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides epochs

    # Optimizer (lower LR than pretraining!)
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    adam_betas: tuple = (0.9, 0.999)

    # Checkpointing
    save_dir: str = "checkpoints/qa"
    save_best_only: bool = True
    eval_interval: int = 50  # Evaluate every N steps

    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "nano_hindi_qa"
    wandb_run_name: Optional[str] = None

    # Generation settings for evaluation
    eval_max_tokens: int = 50
    eval_temperature: float = 0.0  # Greedy for evaluation

    # Format templates
    context_prefix: str = "संदर्भ: "
    question_prefix: str = "\n\nप्रश्न: "
    answer_prefix: str = "\n\nउत्तर:"
    no_answer_text: str = " उत्तर उपलब्ध नहीं है"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def __repr__(self) -> str:
        return (
            f"FinetuneConfig(\n"
            f"  checkpoint={self.checkpoint},\n"
            f"  epochs={self.epochs},\n"
            f"  lr={self.lr},\n"
            f"  batch_size={self.batch_size} × {self.gradient_accumulation_steps} = {self.effective_batch_size},\n"
            f"  max_seq_len={self.max_seq_len}\n"
            f")"
        )
