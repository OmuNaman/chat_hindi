"""
Export nano_hindi PyTorch checkpoint to flat binary format for C inference engine.

Usage:
    python android/tools/export_model.py --checkpoint sft_checkpoints/sft_step600.pt --output android/nano_hindi_250m.bin

Binary format (all little-endian):
    Header: 8 x int32 (32 bytes)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, padded_vocab_size

    Weights: contiguous float32 arrays in this order:
        1. token_embedding_table  (padded_vocab_size x dim)
        2. wq   per layer         (n_layers x dim x dim)
        3. wk   per layer         (n_layers x kv_dim x dim)
        4. wv   per layer         (n_layers x kv_dim x dim)
        5. wo   per layer         (n_layers x dim x dim)
        6. w_fc per layer         (n_layers x hidden_dim x dim)
        7. w_proj per layer       (n_layers x dim x hidden_dim)
        8. resid_lambdas          (n_layers,)
        9. x0_lambdas             (n_layers,)

    Note: PyTorch nn.Linear(in, out) stores weight as (out, in) in row-major.
    This matches C's matmul convention: y[i] = sum_j W[i*n + j] * x[j].
"""

import argparse
import struct
import sys
import os

import torch
import numpy as np


def load_checkpoint(path):
    """Load checkpoint and strip _orig_mod. prefix from keys."""
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Handle various checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip _orig_mod. prefix (added by torch.compile)
    # Only keep tensor values (checkpoint may contain non-tensor metadata)
    cleaned = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_key = k.replace("_orig_mod.", "")
            cleaned[new_key] = v

    return cleaned


def export_model(checkpoint_path, output_path):
    state_dict = load_checkpoint(checkpoint_path)

    # Print all keys for debugging
    print(f"\nFound {len(state_dict)} tensors:")
    for k, v in state_dict.items():
        print(f"  {k}: {list(v.shape)}")

    # Extract config from weight shapes
    wte = state_dict["transformer.wte.weight"]
    padded_vocab_size, dim = wte.shape

    # Find n_layers by counting attention layers
    n_layers = 0
    while f"transformer.h.{n_layers}.attn.c_q.weight" in state_dict:
        n_layers += 1

    wq0 = state_dict["transformer.h.0.attn.c_q.weight"]
    wk0 = state_dict["transformer.h.0.attn.c_k.weight"]
    wfc0 = state_dict["transformer.h.0.mlp.c_fc.weight"]

    n_heads_x_head_dim = wq0.shape[0]  # dim (output of c_q)
    kv_dim = wk0.shape[0]  # n_kv_heads * head_dim
    hidden_dim = wfc0.shape[0]  # 4 * dim

    # Determine head_dim and head counts
    # For 250m: dim=768, n_heads=12, head_dim=64, n_kv_heads=4, kv_dim=256
    # We need to figure out n_heads. We know dim = n_heads * head_dim.
    # From config: possible head_dims are 64 (for all configs).
    # head_dim = dim / n_heads, and kv_dim = n_kv_heads * head_dim
    # So head_dim = kv_dim / n_kv_heads. But we don't know n_kv_heads yet.
    # However, n_heads_x_head_dim = dim, and we can try common head_dims.
    # For this model: head_dim = 64, n_heads = dim / 64, n_kv_heads = kv_dim / 64

    head_dim = 64  # Standard for this model family
    n_heads = dim // head_dim
    n_kv_heads = kv_dim // head_dim

    # The actual vocab_size (before padding) for Sarvam-1
    vocab_size = 68096
    seq_len = 1024

    print(f"\nModel config:")
    print(f"  dim = {dim}")
    print(f"  hidden_dim = {hidden_dim}")
    print(f"  n_layers = {n_layers}")
    print(f"  n_heads = {n_heads}")
    print(f"  n_kv_heads = {n_kv_heads}")
    print(f"  head_dim = {head_dim}")
    print(f"  vocab_size = {vocab_size}")
    print(f"  padded_vocab_size = {padded_vocab_size}")
    print(f"  seq_len = {seq_len}")
    print(f"  kv_dim = {kv_dim}")

    # Verify shapes
    assert dim == 768, f"Expected dim=768, got {dim}"
    assert hidden_dim == 3072, f"Expected hidden_dim=3072, got {hidden_dim}"
    assert n_layers == 32, f"Expected n_layers=32, got {n_layers}"
    assert n_heads == 12, f"Expected n_heads=12, got {n_heads}"
    assert n_kv_heads == 4, f"Expected n_kv_heads=4, got {n_kv_heads}"
    # Note: 68096 is already a multiple of 64, so padded == actual vocab size
    assert padded_vocab_size >= vocab_size, f"Padded vocab {padded_vocab_size} < actual {vocab_size}"

    # Write binary file
    print(f"\nWriting binary to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        # Header: 8 ints
        header = struct.pack("iiiiiiii",
            dim, hidden_dim, n_layers, n_heads,
            n_kv_heads, vocab_size, seq_len, padded_vocab_size
        )
        f.write(header)
        print(f"  Header: {len(header)} bytes")

        # 1. Token embedding table (padded_vocab_size x dim)
        w = wte.float().numpy()
        f.write(w.tobytes())
        print(f"  token_embedding_table: {w.shape} = {w.nbytes:,} bytes")

        # 2. wq - all layers (n_layers x dim x dim)
        total = 0
        for l in range(n_layers):
            w = state_dict[f"transformer.h.{l}.attn.c_q.weight"].float().numpy()
            assert w.shape == (dim, dim), f"wq[{l}] shape {w.shape} != ({dim}, {dim})"
            f.write(w.tobytes())
            total += w.nbytes
        print(f"  wq: ({n_layers} x {dim} x {dim}) = {total:,} bytes")

        # 3. wk - all layers (n_layers x kv_dim x dim)
        total = 0
        for l in range(n_layers):
            w = state_dict[f"transformer.h.{l}.attn.c_k.weight"].float().numpy()
            assert w.shape == (kv_dim, dim), f"wk[{l}] shape {w.shape} != ({kv_dim}, {dim})"
            f.write(w.tobytes())
            total += w.nbytes
        print(f"  wk: ({n_layers} x {kv_dim} x {dim}) = {total:,} bytes")

        # 4. wv - all layers (n_layers x kv_dim x dim)
        total = 0
        for l in range(n_layers):
            w = state_dict[f"transformer.h.{l}.attn.c_v.weight"].float().numpy()
            assert w.shape == (kv_dim, dim), f"wv[{l}] shape {w.shape} != ({kv_dim}, {dim})"
            f.write(w.tobytes())
            total += w.nbytes
        print(f"  wv: ({n_layers} x {kv_dim} x {dim}) = {total:,} bytes")

        # 5. wo - all layers (n_layers x dim x dim)
        total = 0
        for l in range(n_layers):
            w = state_dict[f"transformer.h.{l}.attn.c_proj.weight"].float().numpy()
            assert w.shape == (dim, dim), f"wo[{l}] shape {w.shape} != ({dim}, {dim})"
            f.write(w.tobytes())
            total += w.nbytes
        print(f"  wo: ({n_layers} x {dim} x {dim}) = {total:,} bytes")

        # 6. w_fc - all layers (n_layers x hidden_dim x dim)
        total = 0
        for l in range(n_layers):
            w = state_dict[f"transformer.h.{l}.mlp.c_fc.weight"].float().numpy()
            assert w.shape == (hidden_dim, dim), f"w_fc[{l}] shape {w.shape} != ({hidden_dim}, {dim})"
            f.write(w.tobytes())
            total += w.nbytes
        print(f"  w_fc: ({n_layers} x {hidden_dim} x {dim}) = {total:,} bytes")

        # 7. w_proj - all layers (n_layers x dim x hidden_dim)
        total = 0
        for l in range(n_layers):
            w = state_dict[f"transformer.h.{l}.mlp.c_proj.weight"].float().numpy()
            assert w.shape == (dim, hidden_dim), f"w_proj[{l}] shape {w.shape} != ({dim}, {hidden_dim})"
            f.write(w.tobytes())
            total += w.nbytes
        print(f"  w_proj: ({n_layers} x {dim} x {hidden_dim}) = {total:,} bytes")

        # 8. resid_lambdas (n_layers,)
        w = state_dict["resid_lambdas"].float().numpy()
        assert w.shape == (n_layers,), f"resid_lambdas shape {w.shape}"
        f.write(w.tobytes())
        print(f"  resid_lambdas: {w.shape} = {w.nbytes} bytes, values: {w}")

        # 9. x0_lambdas (n_layers,)
        w = state_dict["x0_lambdas"].float().numpy()
        assert w.shape == (n_layers,), f"x0_lambdas shape {w.shape}"
        f.write(w.tobytes())
        print(f"  x0_lambdas: {w.shape} = {w.nbytes} bytes, values: {w}")

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Output: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

    # Verify expected size
    expected = 32  # header
    expected += padded_vocab_size * dim * 4
    expected += n_layers * dim * dim * 4        # wq
    expected += n_layers * kv_dim * dim * 4     # wk
    expected += n_layers * kv_dim * dim * 4     # wv
    expected += n_layers * dim * dim * 4        # wo
    expected += n_layers * hidden_dim * dim * 4 # w_fc
    expected += n_layers * dim * hidden_dim * 4 # w_proj
    expected += n_layers * 4                    # resid_lambdas
    expected += n_layers * 4                    # x0_lambdas
    assert file_size == expected, f"Size mismatch: {file_size} != {expected}"
    print(f"Size verified: {file_size} == {expected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export nano_hindi checkpoint to C binary format")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="android/nano_hindi_250m.bin",
                        help="Output binary file path")
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)
