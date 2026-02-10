"""
Verify the C binary format by doing a forward pass in Python and comparing.
Loads the exported binary, runs one forward pass, and compares against PyTorch model.
"""
import struct
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT, norm

def load_binary(path):
    """Load the exported binary and return config + weight arrays."""
    with open(path, "rb") as f:
        # Read header
        header = struct.unpack("iiiiiiii", f.read(32))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, padded_vocab_size = header
        print(f"Config: dim={dim} hidden={hidden_dim} layers={n_layers} heads={n_heads} kv_heads={n_kv_heads} vocab={vocab_size} seq={seq_len} padded={padded_vocab_size}")

        head_dim = dim // n_heads
        kv_dim = n_kv_heads * head_dim

        # Read weights
        wte = np.frombuffer(f.read(padded_vocab_size * dim * 4), dtype=np.float32).reshape(padded_vocab_size, dim)
        wq = np.frombuffer(f.read(n_layers * dim * dim * 4), dtype=np.float32).reshape(n_layers, dim, dim)
        wk = np.frombuffer(f.read(n_layers * kv_dim * dim * 4), dtype=np.float32).reshape(n_layers, kv_dim, dim)
        wv = np.frombuffer(f.read(n_layers * kv_dim * dim * 4), dtype=np.float32).reshape(n_layers, kv_dim, dim)
        wo = np.frombuffer(f.read(n_layers * dim * dim * 4), dtype=np.float32).reshape(n_layers, dim, dim)
        w_fc = np.frombuffer(f.read(n_layers * hidden_dim * dim * 4), dtype=np.float32).reshape(n_layers, hidden_dim, dim)
        w_proj = np.frombuffer(f.read(n_layers * dim * hidden_dim * 4), dtype=np.float32).reshape(n_layers, dim, hidden_dim)
        resid_lambdas = np.frombuffer(f.read(n_layers * 4), dtype=np.float32)
        x0_lambdas = np.frombuffer(f.read(n_layers * 4), dtype=np.float32)

    return {
        "config": header,
        "wte": wte, "wq": wq, "wk": wk, "wv": wv, "wo": wo,
        "w_fc": w_fc, "w_proj": w_proj,
        "resid_lambdas": resid_lambdas, "x0_lambdas": x0_lambdas,
    }


def rmsnorm_np(x):
    """RMSNorm with no learnable params."""
    ss = np.mean(x * x) + 1e-6
    return x / np.sqrt(ss)


def forward_one_token_np(weights, token_id, pos):
    """Forward pass for a single token using numpy (matches C engine logic)."""
    cfg = weights["config"]
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, padded_vocab_size = cfg
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    kv_mul = n_heads // n_kv_heads

    # 1. Embedding lookup
    x = weights["wte"][token_id].copy()

    # 2. Post-embed norm
    x = rmsnorm_np(x)

    # 3. Save x0
    x0 = x.copy()

    # We only do layer 0 to compare
    for l in range(1):  # Just first layer for debugging
        # Scalar mixing
        x = weights["resid_lambdas"][l] * x + weights["x0_lambdas"][l] * x0

        # Pre-attn norm
        xb = rmsnorm_np(x)

        # QKV
        q = weights["wq"][l] @ xb  # (dim,)
        k = weights["wk"][l] @ xb  # (kv_dim,)
        v = weights["wv"][l] @ xb  # (kv_dim,)

        print(f"\nLayer {l} after QKV:")
        print(f"  q[:5] = {q[:5]}")
        print(f"  k[:5] = {k[:5]}")

        # RoPE (half-split)
        half = head_dim // 2
        for h in range(n_heads):
            qh = q[h*head_dim:(h+1)*head_dim]
            for i in range(half):
                freq = 1.0 / (10000.0 ** (2*i / head_dim))
                angle = pos * freq
                cos_val = np.cos(angle)
                sin_val = np.sin(angle)
                x1, x2 = qh[i], qh[i + half]
                qh[i] = x1 * cos_val + x2 * sin_val
                qh[i + half] = -x1 * sin_val + x2 * cos_val

        for h in range(n_kv_heads):
            kh = k[h*head_dim:(h+1)*head_dim]
            for i in range(half):
                freq = 1.0 / (10000.0 ** (2*i / head_dim))
                angle = pos * freq
                cos_val = np.cos(angle)
                sin_val = np.sin(angle)
                x1, x2 = kh[i], kh[i + half]
                kh[i] = x1 * cos_val + x2 * sin_val
                kh[i + half] = -x1 * sin_val + x2 * cos_val

        # QK norm (per head)
        for h in range(n_heads):
            q[h*head_dim:(h+1)*head_dim] = rmsnorm_np(q[h*head_dim:(h+1)*head_dim])
        for h in range(n_kv_heads):
            k[h*head_dim:(h+1)*head_dim] = rmsnorm_np(k[h*head_dim:(h+1)*head_dim])

        print(f"  q after rope+norm [:5] = {q[:5]}")
        print(f"  k after rope+norm [:5] = {k[:5]}")

        # Single position attention (pos=0): just self-attention
        # score = q . k / sqrt(d) for each head
        attn_out = np.zeros(dim)
        for h in range(n_heads):
            kv_h = h // kv_mul
            qh = q[h*head_dim:(h+1)*head_dim]
            kh = k[kv_h*head_dim:(kv_h+1)*head_dim]
            score = np.dot(qh, kh) / np.sqrt(head_dim)
            # softmax of single element = 1.0
            vh = v[kv_h*head_dim:(kv_h+1)*head_dim]
            attn_out[h*head_dim:(h+1)*head_dim] = vh

        # Output proj + residual
        x = x + weights["wo"][l] @ attn_out

        # Pre-MLP norm
        xb = rmsnorm_np(x)

        # MLP: ReLU²
        h = weights["w_fc"][l] @ xb
        h = np.maximum(h, 0) ** 2
        x = x + weights["w_proj"][l] @ h

    print(f"\nAfter layer 0: x[:5] = {x[:5]}")
    return x


def forward_pytorch(checkpoint_path, token_id):
    """Forward pass using PyTorch model for comparison."""
    print("\n=== PyTorch Forward ===")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    config = GPTConfig(n_layer=32, n_head=12, n_kv_head=4, n_embd=768,
                       tie_embeddings=True, use_value_embeds=False)
    model = GPT(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    idx = torch.tensor([[token_id]])
    with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        logits = model(idx)

    logits = logits[0, 0]  # (vocab_size,)
    top5 = torch.topk(logits, 5)
    print(f"PyTorch top-5 logits: {list(zip(top5.indices.tolist(), top5.values.tolist()))}")
    return logits


if __name__ == "__main__":
    # Load binary
    bin_weights = load_binary("android/nano_hindi_250m.bin")

    # Test with BOS token (id=1)
    token_id = 1
    print(f"\n=== Numpy Forward (token={token_id}, pos=0) ===")
    x_np = forward_one_token_np(bin_weights, token_id, pos=0)

    # Compare with PyTorch
    logits_pt = forward_pytorch("sft_checkpoints/sft_step600.pt", token_id)
