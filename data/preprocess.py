"""
Preprocess downloaded Hindi data into binary format for training.

Usage:
    python data/preprocess.py --input data/raw/hindi_corpus.txt --output_dir data
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def preprocess_to_binary(
    input_file: str,
    output_dir: str,
    tokenizer_name: str = "sarvamai/sarvam-1",
    val_ratio: float = 0.01,
    chunk_size: int = 10000,
):
    """
    Tokenize text corpus and save as memory-mapped binary files.

    Args:
        input_file: Path to raw text file
        output_dir: Directory to save train.bin and val.bin
        tokenizer_name: HuggingFace tokenizer name
        val_ratio: Fraction of data to use for validation
        chunk_size: Number of documents to process at once
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Use uint32 since vocab > 65535
    dtype = np.uint32

    print(f"Reading input file: {input_file}")

    # First pass: count total tokens
    all_tokens = []
    current_doc = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if line == "<|endoftext|>":
                if current_doc:
                    text = " ".join(current_doc)
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    # Add EOS token between documents
                    if tokenizer.eos_token_id is not None:
                        tokens.append(tokenizer.eos_token_id)
                    all_tokens.extend(tokens)
                    current_doc = []
            else:
                if line:
                    current_doc.append(line)

        # Don't forget the last document
        if current_doc:
            text = " ".join(current_doc)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
            all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    print(f"Total tokens: {total_tokens:,}")

    # Split into train/val
    val_size = int(total_tokens * val_ratio)
    train_size = total_tokens - val_size

    print(f"Train tokens: {train_size:,}")
    print(f"Val tokens: {val_size:,}")

    # Convert to numpy arrays
    all_tokens = np.array(all_tokens, dtype=dtype)

    # Shuffle and split (shuffle at document boundary would be better, but this is simpler)
    # For language modeling, we typically don't shuffle within the corpus
    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]

    # Save as binary files
    train_file = output_path / "train.bin"
    val_file = output_path / "val.bin"

    print(f"Saving train data to {train_file}")
    train_tokens.tofile(train_file)

    print(f"Saving val data to {val_file}")
    val_tokens.tofile(val_file)

    # Save metadata
    meta_file = output_path / "preprocess_meta.txt"
    with open(meta_file, "w") as f:
        f.write(f"tokenizer: {tokenizer_name}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"dtype: {dtype.__name__}\n")
        f.write(f"total_tokens: {total_tokens}\n")
        f.write(f"train_tokens: {train_size}\n")
        f.write(f"val_tokens: {val_size}\n")

    print("\nPreprocessing complete!")
    print(f"  Train file: {train_file} ({train_size:,} tokens)")
    print(f"  Val file: {val_file} ({val_size:,} tokens)")

    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="Preprocess Hindi corpus")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/hindi_corpus.txt",
        help="Input text file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for binary files",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="sarvamai/sarvam-1",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
        help="Fraction of data for validation",
    )

    args = parser.parse_args()

    preprocess_to_binary(
        input_file=args.input,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
