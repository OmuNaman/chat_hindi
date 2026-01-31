import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

EOT = "<|endoftext|>"

# Enable HF tokenizer threading (Rust/Rayon)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def iter_docs(input_file: str):
    """Stream documents separated by <|endoftext|>. Constant memory."""
    current_doc = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == EOT:
                if current_doc:
                    yield " ".join(current_doc)
                    current_doc = []
            elif line:
                current_doc.append(line)
        if current_doc:
            yield " ".join(current_doc)


def encode_batch_to_flat_uint32(tokenizer, batch, eos_id, dtype=np.uint32):
    """Batch tokenize, append EOS per doc, return one flat uint32 array."""
    enc = tokenizer(
        batch,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids_list = enc["input_ids"]

    add_eos = 1 if eos_id is not None else 0
    total = sum(len(ids) + add_eos for ids in ids_list)

    out = np.empty(total, dtype=dtype)
    pos = 0
    for ids in ids_list:
        n = len(ids)
        out[pos:pos + n] = ids
        pos += n
        if eos_id is not None:
            out[pos] = eos_id
            pos += 1

    return out, total


def split_tail_bytes(all_path: Path, train_path: Path, val_path: Path, total_tokens: int, val_ratio: float, dtype=np.uint32):
    """Copy last val_ratio tokens to val.bin, truncate remainder as train.bin."""
    itemsize = np.dtype(dtype).itemsize
    val_tokens = int(total_tokens * val_ratio)
    train_tokens = total_tokens - val_tokens

    total_bytes = total_tokens * itemsize
    val_bytes = val_tokens * itemsize
    train_bytes = total_bytes - val_bytes

    # Copy tail to val.bin
    buf_size = 64 * 1024 * 1024  # 64MB
    with open(all_path, "rb") as fin, open(val_path, "wb") as fout:
        fin.seek(train_bytes)
        remaining = val_bytes
        while remaining > 0:
            chunk = fin.read(min(buf_size, remaining))
            if not chunk:
                raise RuntimeError("Unexpected EOF while copying val tail")
            fout.write(chunk)
            remaining -= len(chunk)

    # Truncate and rename to train.bin
    os.truncate(all_path, train_bytes)
    os.rename(all_path, train_path)

    return train_tokens, val_tokens


def preprocess_to_binary(
    input_file: str,
    output_dir: str,
    tokenizer_name: str = "sarvamai/sarvam-1",
    val_ratio: float = 0.01,
    batch_size: int = 4096,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id = tokenizer.eos_token_id
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token id: {eos_id}")
    print(f"Batch size: {batch_size} docs")
    print(f"RAYON_NUM_THREADS: {os.environ.get('RAYON_NUM_THREADS', 'all cores')}")

    dtype = np.uint32
    all_path = out_dir / "all_tokens.tmp"
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    total_tokens = 0
    total_docs = 0
    batch = []

    print("Tokenizing and streaming to disk...")
    with open(all_path, "wb") as out_f:
        for doc in tqdm(iter_docs(input_file), desc="Tokenizing", unit=" docs"):
            batch.append(doc)
            if len(batch) >= batch_size:
                arr, n = encode_batch_to_flat_uint32(tokenizer, batch, eos_id, dtype=dtype)
                arr.tofile(out_f)  # ONE write per batch
                total_tokens += n
                total_docs += len(batch)
                batch.clear()

        if batch:
            arr, n = encode_batch_to_flat_uint32(tokenizer, batch, eos_id, dtype=dtype)
            arr.tofile(out_f)
            total_tokens += n
            total_docs += len(batch)
            batch.clear()

    print(f"Total documents: {total_docs:,}")
    print(f"Total tokens: {total_tokens:,}")

    train_tokens, val_tokens = split_tail_bytes(
        all_path=all_path,
        train_path=train_path,
        val_path=val_path,
        total_tokens=total_tokens,
        val_ratio=val_ratio,
        dtype=dtype,
    )

    meta = out_dir / "preprocess_meta.txt"
    with open(meta, "w", encoding="utf-8") as f:
        f.write(f"tokenizer: {tokenizer_name}\n")
        f.write(f"vocab_size: {tokenizer.vocab_size}\n")
        f.write(f"dtype: {dtype.__name__}\n")
        f.write(f"total_tokens: {total_tokens}\n")
        f.write(f"train_tokens: {train_tokens}\n")
        f.write(f"val_tokens: {val_tokens}\n")
        f.write(f"val_ratio: {val_ratio}\n")
        f.write(f"batch_size: {batch_size}\n")

    print("\nPreprocessing complete!")
    print(f"  Train file: {train_path} ({train_tokens:,} tokens)")
    print(f"  Val file:   {val_path} ({val_tokens:,} tokens)")
    print(f"  Meta:       {meta}")

    return train_path, val_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/raw/hindi_corpus.txt")
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--tokenizer", type=str, default="sarvamai/sarvam-1")
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=4096)
    args = p.parse_args()

    preprocess_to_binary(
        input_file=args.input,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()