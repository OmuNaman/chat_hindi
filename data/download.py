"""
Download Hindi data from Sangraha dataset.

Usage:
    python data/download.py --output_dir data/raw --max_tokens 600000000
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a text string."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def download_sangraha_hindi(
    output_dir: str,
    max_tokens: int = 600_000_000,
    tokenizer_name: str = "sarvamai/sarvam-1",
):
    """
    Download Hindi verified subset from Sangraha.

    The dataset is streamed and saved to disk to avoid memory issues.
    We count tokens and stop when we reach max_tokens.

    Args:
        output_dir: Directory to save the raw text files
        max_tokens: Maximum number of tokens to download (Chinchilla scaling)
        tokenizer_name: Tokenizer to use for counting
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Streaming Sangraha Hindi verified subset...")
    print(f"Target: {max_tokens:,} tokens")

    # Stream the dataset
    dataset = load_dataset(
        "ai4bharat/sangraha",
        data_dir="verified/hin",
        split="train",
        streaming=True,
    )

    total_tokens = 0
    total_docs = 0
    output_file = output_path / "hindi_corpus.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(dataset, desc="Downloading", unit=" docs")

        for doc in pbar:
            text = doc.get("text", "")
            if not text or len(text.strip()) == 0:
                continue

            # Count tokens
            n_tokens = count_tokens(text, tokenizer)
            total_tokens += n_tokens
            total_docs += 1

            # Write to file with document separator
            f.write(text.strip())
            f.write("\n<|endoftext|>\n")

            # Update progress
            pbar.set_postfix(
                tokens=f"{total_tokens / 1e6:.1f}M",
                docs=total_docs,
            )

            # Check if we've reached our target
            if total_tokens >= max_tokens:
                print(f"\nReached target of {max_tokens:,} tokens")
                break

    print(f"\nDownload complete!")
    print(f"  Total documents: {total_docs:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Output file: {output_file}")

    # Save metadata
    meta_file = output_path / "metadata.txt"
    with open(meta_file, "w") as f:
        f.write(f"total_documents: {total_docs}\n")
        f.write(f"total_tokens: {total_tokens}\n")
        f.write(f"tokenizer: {tokenizer_name}\n")
        f.write(f"source: ai4bharat/sangraha verified/hin\n")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download Sangraha Hindi dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save raw data",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=600_000_000,
        help="Maximum tokens to download (default: 600M for ~30M model)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="sarvamai/sarvam-1",
        help="Tokenizer for counting tokens",
    )

    args = parser.parse_args()

    download_sangraha_hindi(
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        tokenizer_name=args.tokenizer,
    )


if __name__ == "__main__":
    main()
