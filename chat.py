"""
Minimalist chat UI for nano_hindi SFT model.

Usage:
    python chat.py --checkpoint sft_checkpoints/sft_step600.pt
    python chat.py --checkpoint sft_checkpoints/sft_step600.pt --port 8080
    python chat.py --checkpoint sft_checkpoints/sft_step600.pt --no-compile
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, render_template, request
from transformers import AutoTokenizer

from huggingface_hub import hf_hub_download

from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT
from nano_hindi.tokenizer import ASSISTANT_MARKER, USER_MARKER

app = Flask(__name__)

# Globals (loaded once at startup)
MODEL = None
TOKENIZER = None
DEVICE = None
KV_CACHE = None  # Pre-allocated, reused across requests
MODEL_CONFIG = None
HISTORY_DIR = Path("chat_history")


# --- KV Cache ---

class KVCache:
    """Pre-allocated KV cache for fast autoregressive generation."""

    def __init__(self, batch_size, max_seq_len, n_layers, n_kv_head, head_dim, device, dtype=torch.bfloat16):
        self.n_layers = n_layers
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.k_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_kv_head, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_kv_head, head_dim, device=device, dtype=dtype)

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def advance(self, n):
        self.cache_seqlens += n

    def reset(self):
        self.cache_seqlens.fill_(0)


# --- Model loading ---

def load_model(checkpoint_path: str, device: str = "cuda", compile_model: bool = True):
    """Load model from checkpoint, optionally compile."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint["model_config"]
    config = GPTConfig(**{k: v for k, v in config_dict.items() if k != "head_dim"})
    print(f"Model config: {config}")

    model = GPT(config).to(device)

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Model loaded from step {checkpoint['step']}")

    if compile_model and device == "cuda":
        print("Compiling model with torch.compile (this takes a minute on first run)...")
        model = torch.compile(model)
        print("Model compiled.")

    return model, config


# --- Context building ---

def build_context_ids(messages):
    """Convert messages list to token IDs for model input."""
    bos_id = TOKENIZER.bos_token_id
    context_ids = [bos_id]

    for msg in messages:
        if msg["role"] == "user":
            text = USER_MARKER + msg["content"] + "\n\n"
            context_ids.extend(TOKENIZER.encode(text, add_special_tokens=False))
        elif msg["role"] == "assistant":
            text = ASSISTANT_MARKER + msg["content"] + "\n\n"
            context_ids.extend(TOKENIZER.encode(text, add_special_tokens=False))

    # Prime assistant generation
    marker_ids = TOKENIZER.encode(ASSISTANT_MARKER, add_special_tokens=False)
    context_ids.extend(marker_ids)

    # Truncate to 900 tokens (seq_len=1024, leave room for generation)
    if len(context_ids) > 900:
        context_ids = [bos_id] + context_ids[-899:]

    return context_ids


# --- Generation with KV cache ---

@torch.inference_mode()
def generate_stream(messages, temperature=0.8, top_k=50, max_tokens=512):
    """Generator that yields SSE events with token deltas. Uses KV cache for speed."""
    context_ids = build_context_ids(messages)
    eos_id = TOKENIZER.eos_token_id
    seed = int(time.time() * 1000) % (2**31)

    # Get raw model (unwrap compile wrapper)
    raw = MODEL
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod

    dev = raw.get_device()
    dev_type = dev.type if hasattr(dev, "type") else "cuda"

    rng = torch.Generator(device=dev)
    rng.manual_seed(seed)

    # Reset pre-allocated KV cache
    KV_CACHE.reset()

    # Prefill: process entire prompt at once
    prompt_tensor = torch.tensor([context_ids], dtype=torch.long, device=dev)
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
        logits = MODEL(prompt_tensor, kv_cache=KV_CACHE)
    logits = logits[:, -1, :]  # last position

    # Sample first token
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
    if temperature > 0:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=rng)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    token_id = next_token.item()
    generated_tokens = []
    prev_text = ""

    # Yield first token
    if token_id != eos_id:
        generated_tokens.append(token_id)
        full_text = TOKENIZER.decode(generated_tokens)
        if full_text:
            yield f"data: {json.dumps({'token': full_text})}\n\n"
            prev_text = full_text

    # Decode loop: one token at a time with KV cache
    max_seq_len = MODEL_CONFIG.sequence_len
    for _ in range(1, max_tokens):
        if token_id == eos_id:
            break
        if KV_CACHE.get_pos() >= max_seq_len:
            break

        input_tensor = next_token  # (1, 1)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
            logits = MODEL(input_tensor, kv_cache=KV_CACHE)
        logits = logits[:, -1, :]

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        if token_id == eos_id:
            break

        generated_tokens.append(token_id)
        full_text = TOKENIZER.decode(generated_tokens)
        delta = full_text[len(prev_text):]
        if delta:
            yield f"data: {json.dumps({'token': delta})}\n\n"
            prev_text = full_text

    yield f"data: {json.dumps({'done': True})}\n\n"


# --- Routes ---

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None, "device": str(DEVICE)})


@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.8)
    top_k = data.get("top_k", 50)
    max_tokens = data.get("max_tokens", 512)

    return Response(
        generate_stream(messages, temperature, top_k, max_tokens),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/conversations", methods=["GET"])
def list_conversations():
    conversations = []
    if HISTORY_DIR.exists():
        for f in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                conversations.append({
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    return jsonify(conversations)


@app.route("/conversations/<conv_id>", methods=["GET"])
def get_conversation(conv_id):
    filepath = HISTORY_DIR / f"{conv_id}.json"
    if not filepath.exists():
        return jsonify({"error": "Not found"}), 404
    data = json.loads(filepath.read_text(encoding="utf-8"))
    return jsonify(data)


@app.route("/conversations", methods=["POST"])
def save_conversation():
    data = request.json
    conv_id = data.get("id") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    messages = data.get("messages", [])

    # Auto-generate title from first user message
    title = "Untitled"
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:50]
            break

    conv_data = {
        "id": conv_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }

    HISTORY_DIR.mkdir(exist_ok=True)
    filepath = HISTORY_DIR / f"{conv_id}.json"
    filepath.write_text(json.dumps(conv_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"id": conv_id, "title": title})


@app.route("/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    filepath = HISTORY_DIR / f"{conv_id}.json"
    if filepath.exists():
        filepath.unlink()
    return jsonify({"ok": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano_hindi Chat UI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SFT checkpoint (auto-downloads from HF if not provided)")
    parser.add_argument("--hf-repo", type=str, default="omunaman/nano-hindi-sft", help="HuggingFace repo for auto-download")
    parser.add_argument("--hf-file", type=str, default="sft_step600.pt", help="Checkpoint filename in HF repo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (Linux only, needs Triton)")
    args = parser.parse_args()

    # Resolve checkpoint: use local file if provided, otherwise download from HF Hub
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt_path = args.checkpoint
    else:
        print(f"Downloading model from {args.hf_repo}/{args.hf_file}...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)
        print(f"Model cached at: {ckpt_path}")

    DEVICE = args.device
    MODEL, MODEL_CONFIG = load_model(ckpt_path, args.device, compile_model=args.compile)
    TOKENIZER = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

    # Pre-allocate KV cache (batch_size=1 for single-user chat)
    KV_CACHE = KVCache(
        batch_size=1,
        max_seq_len=MODEL_CONFIG.sequence_len,
        n_layers=MODEL_CONFIG.n_layer,
        n_kv_head=MODEL_CONFIG.n_kv_head,
        head_dim=MODEL_CONFIG.head_dim,
        device=torch.device(args.device),
    )
    print(f"KV cache allocated: {MODEL_CONFIG.n_layer} layers, seq_len={MODEL_CONFIG.sequence_len}")

    # Warmup: run one forward pass to trigger torch.compile and CUDA kernels
    if args.device == "cuda":
        print("Warming up model...")
        warmup_ids = torch.tensor([[TOKENIZER.bos_token_id]], device=args.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            MODEL(warmup_ids)
        KV_CACHE.reset()
        print("Warmup complete.")

    print(f"\nStarting chat UI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
