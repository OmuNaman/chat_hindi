"""
nano_hindi: A small Hindi language model trained from scratch.
Uses Sarvam tokenizer + Sangraha dataset + modern GPT architecture.
"""

from .config import GPTConfig, TrainConfig
from .model import GPT

__version__ = "0.1.0"
