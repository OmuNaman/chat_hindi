"""
Export Sarvam-1 tokenizer to binary format for C inference engine.

Usage:
    python android/tools/export_tokenizer.py --output android/app/src/main/res/raw/tokenizer.bin

Binary format (all little-endian):
    max_token_length: int32 (max UTF-8 byte length of any token piece)
    For each token (0 to vocab_size-1):
        score: float32 (BPE merge score, used for encoding priority)
        len: int32 (byte length of the piece string)
        piece: bytes[len] (UTF-8 encoded token piece)

This format is compatible with the llama2.c tokenizer approach.
"""

import argparse
import os
import struct
import sys


def export_tokenizer(output_path):
    # Try SentencePiece first (more reliable for scores)
    try:
        import sentencepiece as spm
        use_sp = True
    except ImportError:
        use_sp = False
        print("Warning: sentencepiece not installed, falling back to HuggingFace tokenizer")

    if use_sp:
        export_with_sentencepiece(output_path)
    else:
        export_with_hf(output_path)


def export_with_sentencepiece(output_path):
    """Export using SentencePiece directly for accurate scores."""
    import sentencepiece as spm
    from transformers import AutoTokenizer

    print("Loading Sarvam-1 tokenizer...")
    hf_tok = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

    # Find the SentencePiece model file
    sp = spm.SentencePieceProcessor()
    sp_model_path = None

    # Check tokenizer files directory
    if hasattr(hf_tok, 'vocab_file') and hf_tok.vocab_file:
        sp_model_path = hf_tok.vocab_file
    else:
        # Look in the cache directory
        import glob
        cache_dir = os.path.dirname(hf_tok.name_or_path) if os.path.exists(hf_tok.name_or_path or "") else None
        if cache_dir is None:
            # Try HF cache
            from huggingface_hub import hf_hub_download
            sp_model_path = hf_hub_download(repo_id="sarvamai/sarvam-1", filename="tokenizer.model")

    if sp_model_path and os.path.exists(sp_model_path):
        print(f"Loading SentencePiece model from {sp_model_path}")
        sp.Load(sp_model_path)
    else:
        # Fall back: extract from HF tokenizer if it wraps SP
        if hasattr(hf_tok, 'sp_model'):
            sp = hf_tok.sp_model
        else:
            print("Cannot find SentencePiece model, falling back to HF method")
            export_with_hf(output_path)
            return

    vocab_size = sp.GetPieceSize()
    print(f"Vocab size: {vocab_size}")
    assert vocab_size == 68096, f"Expected 68096, got {vocab_size}"

    # Collect all pieces and scores
    pieces = []
    max_token_length = 0
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)

        # SentencePiece uses special unicode replacements
        # The ▁ character (U+2581) represents a space in SentencePiece
        piece_bytes = piece.encode("utf-8")
        max_token_length = max(max_token_length, len(piece_bytes))
        pieces.append((piece_bytes, score))

    print(f"Max token length: {max_token_length} bytes")

    # Write binary
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("i", max_token_length))

        for i, (piece_bytes, score) in enumerate(pieces):
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", len(piece_bytes)))
            f.write(piece_bytes)

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Output: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    # Print some sample tokens for verification
    print("\nSample tokens:")
    for i in [0, 1, 2, 3, 100, 1000, 10000, 50000]:
        if i < vocab_size:
            score = sp.GetScore(i)
            piece_bytes = pieces[i][0]
            print(f"  [{i}] score={score:.4f} len={len(piece_bytes)} bytes={list(piece_bytes[:20])}")


def export_with_hf(output_path):
    """Export using HuggingFace tokenizer (fallback, scores will be approximated)."""
    from transformers import AutoTokenizer

    print("Loading Sarvam-1 tokenizer via HuggingFace...")
    tok = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

    vocab_size = tok.vocab_size
    print(f"Vocab size: {vocab_size}")
    assert vocab_size == 68096, f"Expected 68096, got {vocab_size}"

    # Build id -> piece mapping
    # HF tokenizer.get_vocab() returns piece -> id
    vocab = tok.get_vocab()
    id_to_piece = {v: k for k, v in vocab.items()}

    # Collect pieces
    pieces = []
    max_token_length = 0
    for i in range(vocab_size):
        if i in id_to_piece:
            piece = id_to_piece[i]
        else:
            # Fallback: decode single token
            try:
                piece = tok.decode([i])
            except Exception:
                piece = ""

        piece_bytes = piece.encode("utf-8")
        max_token_length = max(max_token_length, len(piece_bytes))

        # Approximate score: use negative index as proxy (lower index = higher priority)
        # Special tokens get very low scores
        if i < 3:  # <unk>, <s>, </s>
            score = -1000.0
        else:
            score = -float(i)  # Lower index = higher merge priority (rough approximation)

        pieces.append((piece_bytes, score))

    print(f"Max token length: {max_token_length} bytes")

    # Write binary
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("i", max_token_length))

        for i, (piece_bytes, score) in enumerate(pieces):
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", len(piece_bytes)))
            f.write(piece_bytes)

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Output: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print("\nWARNING: Scores are approximated (sentencepiece not available).")
    print("Encoding quality may be degraded. Install sentencepiece for accurate scores.")

    # Print sample tokens
    print("\nSample tokens:")
    for i in [0, 1, 2, 3, 100, 1000, 10000, 50000]:
        if i < vocab_size:
            piece = pieces[i][0].decode("utf-8", errors="replace")
            score = pieces[i][1]
            print(f"  [{i}] score={score:.4f} piece='{piece}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Sarvam-1 tokenizer to binary format")
    parser.add_argument("--output", type=str,
                        default="android/app/src/main/res/raw/tokenizer.bin",
                        help="Output binary file path")
    args = parser.parse_args()
    export_tokenizer(args.output)
