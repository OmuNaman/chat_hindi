"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs, MPS, and CPU.

Usage:
    from nano_hindi.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace


def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper+ GPU)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:  # Hopper is sm90
            return None
        import os

        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
_override_impl = None

# Check if enable_gqa is supported (PyTorch 2.5+)
_SDPA_SUPPORTS_GQA = False
try:
    import inspect
    sig = inspect.signature(F.scaled_dot_product_attention)
    _SDPA_SUPPORTS_GQA = 'enable_gqa' in sig.parameters
except Exception:
    pass


def _use_fa3():
    """Determine whether to use FA3 based on availability and override."""
    if _override_impl == "fa3":
        assert HAS_FA3, "Cannot override to FA3: not available"
        return True
    if _override_impl == "sdpa":
        return False
    return HAS_FA3


def _repeat_kv(x, n_rep):
    """Repeat KV heads to match Q heads for GQA when enable_gqa not supported.

    x: (B, H_kv, T, D) -> (B, H_kv * n_rep, T, D)
    """
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)


def _sdpa_attention(q, k, v, window_size, n_rep=1):
    """SDPA attention with sliding window support. q, k, v are (B, H, T, D)."""
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Handle GQA: either use enable_gqa or repeat KV heads
    n_heads_q = q.size(1)
    n_heads_kv = k.size(1)

    if n_heads_q != n_heads_kv:
        if _SDPA_SUPPORTS_GQA:
            # Use native GQA support (PyTorch 2.5+)
            enable_gqa = True
        else:
            # Manually repeat KV heads
            n_rep = n_heads_q // n_heads_kv
            k = _repeat_kv(k, n_rep)
            v = _repeat_kv(v, n_rep)
            enable_gqa = False
    else:
        enable_gqa = False

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        if _SDPA_SUPPORTS_GQA and enable_gqa:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Single token generation
    if Tq == 1:
        if _SDPA_SUPPORTS_GQA and enable_gqa:
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Need explicit mask
    device = q.device
    if Tq == Tk:
        mask = torch.tril(torch.ones(Tq, Tk, device=device, dtype=torch.bool))
        if window > 0 and window < Tq:
            row_idx = torch.arange(Tq, device=device).unsqueeze(1)
            col_idx = torch.arange(Tk, device=device).unsqueeze(0)
            mask = mask & ((row_idx - col_idx) <= window)
    else:
        prefix_len = Tk - Tq
        mask = torch.zeros(Tq, Tk, device=device, dtype=torch.bool)
        mask[:, :prefix_len] = True
        mask[:, prefix_len:] = torch.tril(
            torch.ones(Tq, Tq, device=device, dtype=torch.bool)
        )

    if _SDPA_SUPPORTS_GQA and enable_gqa:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    else:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = _sdpa_attention(q, k, v, window_size)
    return y.transpose(1, 2)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    causal=False,
    window_size=(-1, -1),
):
    """
    Flash Attention with KV cache for inference.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors
        k, v: New keys/values to insert
        cache_seqlens: Current position in cache
        causal: Whether to use causal masking
        window_size: (left, right) sliding window

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    # SDPA fallback
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()

    if k is not None and v is not None:
        k_cache[:, pos : pos + T_new, :, :] = k
        v_cache[:, pos : pos + T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size)

    return y_sdpa.transpose(1, 2)


# Export module interface
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
