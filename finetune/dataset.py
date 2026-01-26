"""
IndicQA dataset loading and preprocessing for nano_hindi fine-tuning.

Dataset: https://huggingface.co/datasets/ai4bharat/IndicQA
Format: SQuAD-style with context, question, answer triplets.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import FinetuneConfig


@dataclass
class QAExample:
    """A single QA example."""
    context: str
    question: str
    answer: str
    category: str  # "SHORT" or "NO"
    answer_start: Optional[int] = None


def load_indicqa(json_path: str) -> List[QAExample]:
    """
    Load IndicQA dataset from JSON file.

    Expected format:
    {
        "version": 1.0,
        "data": [
            {
                "paragraphs": [
                    {
                        "context": "...",
                        "qas": [
                            {
                                "question": "...",
                                "category": "SHORT" or "NO",
                                "answers": [{"text": "...", "answer_start": 123}]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                question = qa["question"]
                category = qa.get("category", "SHORT")

                # Get answer text
                if category == "NO" or not qa["answers"] or not qa["answers"][0]["text"]:
                    answer = ""
                    answer_start = None
                else:
                    answer = qa["answers"][0]["text"]
                    answer_start = qa["answers"][0].get("answer_start")

                examples.append(QAExample(
                    context=context,
                    question=question,
                    answer=answer,
                    category=category,
                    answer_start=answer_start,
                ))

    return examples


def download_indicqa(output_dir: str = "data/qa") -> str:
    """
    Download IndicQA Hindi dataset from HuggingFace.

    Returns path to downloaded JSON file.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "indicqa.hi.json"

    if output_path.exists():
        print(f"IndicQA already exists at {output_path}")
        return str(output_path)

    print("Downloading IndicQA Hindi dataset...")
    dataset = load_dataset("ai4bharat/IndicQA", "indicqa.hi", trust_remote_code=True)

    # Convert to IndicQA JSON format
    data = {"version": 1.0, "data": []}

    for split in ["train", "validation"]:
        if split not in dataset:
            continue

        for example in dataset[split]:
            article = {
                "paragraphs": [
                    {
                        "context": example["context"],
                        "qas": [
                            {
                                "question": example["question"],
                                "category": "NO" if not example["answers"]["text"] else "SHORT",
                                "answers": [
                                    {
                                        "text": example["answers"]["text"][0] if example["answers"]["text"] else "",
                                        "answer_start": example["answers"]["answer_start"][0] if example["answers"]["answer_start"] else None,
                                    }
                                ] if example["answers"]["text"] else [],
                            }
                        ],
                    }
                ]
            }
            data["data"].append(article)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved IndicQA to {output_path}")
    print(f"Total examples: {len(data['data'])}")

    return str(output_path)


class IndicQADataset(Dataset):
    """
    PyTorch Dataset for IndicQA fine-tuning.

    Converts QA examples to generative format:
    Input:  "संदर्भ: {context}\n\nप्रश्न: {question}\n\nउत्तर:"
    Target: " {answer}</s>"
    """

    def __init__(
        self,
        examples: List[QAExample],
        tokenizer,
        config: FinetuneConfig,
        is_train: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train

        # Get EOS token
        self.eos_token = tokenizer.eos_token or "</s>"
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")

    def __len__(self) -> int:
        return len(self.examples)

    def format_example(self, example: QAExample) -> Tuple[str, str]:
        """Format a QA example into input/target strings."""
        # Input: context + question + answer prompt
        input_text = (
            f"{self.config.context_prefix}{example.context}"
            f"{self.config.question_prefix}{example.question}"
            f"{self.config.answer_prefix}"
        )

        # Target: answer + EOS
        # Note: If filter_unanswerable=True, we won't have "NO" category examples
        if example.answer:
            target_text = f" {example.answer}{self.eos_token}"
        else:
            target_text = f"{self.config.no_answer_text}{self.eos_token}"

        return input_text, target_text

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        input_text, target_text = self.format_example(example)

        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

        # Tokenize target
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

        # Combine for language modeling: input + target
        full_ids = input_ids + target_ids

        # Truncate if needed (keep end for answer)
        if len(full_ids) > self.config.max_seq_len:
            # Calculate how much to truncate from context
            overflow = len(full_ids) - self.config.max_seq_len

            # Truncate input_ids (context), keep target intact
            input_ids = input_ids[:-overflow] if overflow < len(input_ids) else input_ids[:10]
            full_ids = input_ids + target_ids

            # Final truncation if still too long
            if len(full_ids) > self.config.max_seq_len:
                full_ids = full_ids[:self.config.max_seq_len]

        # Create labels: -100 for input tokens (don't compute loss)
        input_len = len(input_ids)
        labels = [-100] * input_len + target_ids

        # Ensure same length
        labels = labels[:len(full_ids)]

        # Pad to max_seq_len
        pad_len = self.config.max_seq_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id or 0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_dataloaders(
    config: FinetuneConfig,
    tokenizer=None,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[QAExample]]:
    """
    Create train and validation dataloaders.

    Returns:
        train_loader, val_loader, val_examples (for evaluation)
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Load or download data
    if not Path(config.data_path).exists():
        print(f"Data not found at {config.data_path}, downloading...")
        download_indicqa(str(Path(config.data_path).parent))

    # Load examples
    examples = load_indicqa(config.data_path)
    print(f"Loaded {len(examples)} QA examples")

    # Count categories
    short_count = sum(1 for e in examples if e.category == "SHORT")
    no_count = sum(1 for e in examples if e.category == "NO")
    print(f"  SHORT (answerable): {short_count}")
    print(f"  NO (unanswerable): {no_count}")

    # Filter out unanswerable questions (category="NO")
    if config.filter_unanswerable:
        examples = [e for e in examples if e.category != "NO" and e.answer]
        print(f"After filtering unanswerable: {len(examples)} examples")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(examples)

    split_idx = int(len(examples) * config.train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Create datasets
    train_dataset = IndicQADataset(train_examples, tokenizer, config, is_train=True)
    val_dataset = IndicQADataset(val_examples, tokenizer, config, is_train=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, val_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare IndicQA dataset")
    parser.add_argument("--output_dir", type=str, default="data/qa", help="Output directory")
    args = parser.parse_args()

    # Download dataset
    output_path = download_indicqa(args.output_dir)

    # Test loading
    examples = load_indicqa(output_path)
    print(f"\nLoaded {len(examples)} examples")

    # Show a few examples
    print("\n--- Sample Examples ---")
    for i, ex in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Context: {ex.context[:100]}...")
        print(f"  Question: {ex.question}")
        print(f"  Answer: {ex.answer if ex.answer else '(no answer)'}")
        print(f"  Category: {ex.category}")
