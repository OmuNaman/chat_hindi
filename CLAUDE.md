# nano_hindi - Hindi Language Model Training Project

## Project Overview

Training a ~25M parameter Hindi language model from scratch using:
- **Tokenizer**: Sarvam-1 (68,096 vocab, Hindi-optimized, fertility ~1.4)
- **Dataset**: Sangraha Hindi verified subset from AI4Bharat
- **Architecture**: Modern GPT with RoPE, GQA, ReLU², sliding window attention
- **Optimizer**: Muon (for weights) + AdamW (for embeddings)
- **Scaling**: Chinchilla 20:1 ratio (25M params × 20 = 500M tokens)

---

## What Has Been Done ✅

### 1. Project Structure Created
```
d:\Coding_Workspace\nano_hindi\
├── nano_hindi/                 # Main package
│   ├── __init__.py
│   ├── config.py               # GPTConfig + TrainConfig dataclasses
│   ├── model.py                # Full GPT implementation
│   ├── muon.py                 # Muon optimizer (+ DistMuon for DDP)
│   ├── flash_attention.py      # FA3/SDPA unified interface
│   ├── eval.py                 # BPB evaluation
│   └── common.py               # Distributed training utils
├── data/
│   ├── __init__.py
│   ├── download.py             # Download Sangraha Hindi
│   └── preprocess.py           # Tokenize to binary format
├── checkpoints/                # Will store model checkpoints
├── train.py                    # Main training script
├── generate.py                 # Inference script
├── requirements.txt            # Python dependencies
├── setup_env.bat               # Windows environment setup
└── run_training.bat            # Quick training launcher
```

### 2. Model Architecture (nano_hindi/model.py)

Implemented modern GPT architecture with these features:
- **Rotary Position Embeddings (RoPE)**: No learned positional embeddings
- **QK Normalization**: RMSNorm applied to Q and K before attention
- **Group Query Attention (GQA)**: Fewer KV heads than Q heads (3:1 ratio)
- **ReLU² Activation**: `F.relu(x).square()` in MLP
- **Sliding Window Attention**: Pattern "SSSL" (3 short, 1 long window)
- **Value Embeddings (ResFormer)**: Alternating layers get value embeddings
- **Per-layer Scalars**: `resid_lambdas` and `x0_lambdas` for residual mixing
- **Untied Embeddings**: Separate token embedding and lm_head
- **No Bias**: All linear layers have `bias=False`
- **Logit Softcap**: `15 * tanh(logits / 15)` for stability

### 3. Model Configurations (nano_hindi/config.py)

Three preset sizes available:

| Config | Params | Layers | d_model | Heads | KV Heads | Tokens (20:1) |
|--------|--------|--------|---------|-------|----------|---------------|
| `25m`  | ~25M   | 8      | 384     | 6     | 2        | 500M          |
| `40m`  | ~40M   | 10     | 512     | 8     | 2        | 800M          |
| `50m`  | ~50M   | 12     | 576     | 8     | 2        | 1B            |

**GPTConfig defaults:**
```python
vocab_size = 68096      # Sarvam tokenizer
n_layer = 8
n_head = 6
n_kv_head = 2           # GQA 3:1 ratio
n_embd = 384
sequence_len = 1024
window_pattern = "SSSL"
```

**TrainConfig defaults:**
```python
batch_size = 64
gradient_accumulation_steps = 4
total_tokens = 500_000_000
muon_lr = 0.02
adamw_embedding_lr = 0.2
adamw_unembedding_lr = 0.004
warmup_steps = 1000
checkpoint_interval = 1000
eval_interval = 500
inference_interval = 1000
```

### 4. Muon Optimizer (nano_hindi/muon.py)

Implemented Muon optimizer with:
- **Polar Express orthogonalization**: 5 Newton-Schulz iterations
- **Nesterov momentum**: Default 0.95
- **Variance reduction**: Second moment buffer
- **Cautious weight decay**: Only decays where update and weight agree
- **Fused kernel**: `@torch.compile` for efficiency
- **DistMuon**: Distributed version with reduce_scatter/all_gather

### 5. Flash Attention Wrapper (nano_hindi/flash_attention.py)

Unified interface that:
- Uses Flash Attention 3 on Hopper+ GPUs (sm90+)
- Falls back to PyTorch SDPA on older GPUs
- Supports sliding window attention
- Handles KV cache for inference

### 6. Data Pipeline (data/download.py, data/preprocess.py)

**Download script:**
- Streams from `ai4bharat/sangraha` dataset
- Specifically downloads `verified/hin` subset
- Counts tokens using Sarvam tokenizer
- Stops at `max_tokens` (default 600M)
- Saves raw text with `<|endoftext|>` separators

**Preprocess script:**
- Tokenizes with Sarvam tokenizer (`sarvamai/sarvam-1`)
- Saves as memory-mapped binary (uint32, since vocab > 65535)
- Splits into train.bin (99%) and val.bin (1%)

### 7. Training Script (train.py)

Features:
- **Dual optimizer**: Muon for 2D weights, AdamW for embeddings/scalars
- **Cosine LR schedule**: With warmup
- **Gradient accumulation**: Effective batch = 64 × 4 = 256
- **Checkpointing**: Every 1000 steps, keeps last 3
- **Evaluation**: Loss on validation set every 500 steps
- **Inference samples**: Generates Hindi text every 1000 steps
- **Wandb logging**: Optional, tracks loss/lr/samples
- **torch.compile**: For faster training
- **DDP support**: For multi-GPU training

### 8. Inference Script (generate.py)

Features:
- Load model from checkpoint
- Single prompt generation
- Interactive mode with Hindi prompts
- Configurable temperature, top_k, max_tokens
- Demo mode with preset Hindi prompts

### 9. Evaluation (nano_hindi/eval.py)

- **Bits Per Byte (BPB)**: Tokenization-independent metric
- **compute_token_bytes()**: Maps token IDs to UTF-8 byte counts
- **evaluate_loss()**: Simple cross-entropy loss

---

## What Needs To Be Done 📋

### Step 1: Setup Environment
```bash
# Run the setup script (creates venv, installs dependencies)
setup_env.bat

# Or manually:
python -m venv venv
venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2: Download Hindi Data
```bash
# Download ~500M tokens from Sangraha Hindi verified subset
python data/download.py --output_dir data/raw --max_tokens 500000000

# This will create:
# - data/raw/hindi_corpus.txt (~several GB)
# - data/raw/metadata.txt
```

**Expected output:**
- ~500M tokens
- Source: https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/hin
- Format: Text with `<|endoftext|>` document separators

### Step 3: Preprocess Data
```bash
# Tokenize and create binary files
python data/preprocess.py --input data/raw/hindi_corpus.txt --output_dir data

# This will create:
# - data/train.bin (~1-2 GB)
# - data/val.bin (~10-20 MB)
# - data/preprocess_meta.txt
```

### Step 4: Train Model
```bash
# Basic training
python train.py --config 25m --total_tokens 500000000

# With wandb logging
python train.py --config 25m --total_tokens 500000000 --wandb

# Resume from checkpoint
python train.py --config 25m --resume checkpoints/checkpoint_step5000.pt
```

**Training will:**
- Create checkpoints in `checkpoints/` every 1000 steps
- Log to console every 10 steps
- Evaluate on val set every 500 steps
- Generate Hindi samples every 1000 steps
- Total steps: ~7,630 (500M tokens / 65K tokens per step)

### Step 5: Generate Text
```bash
# Interactive mode
python generate.py --checkpoint checkpoints/checkpoint_step7000.pt --interactive

# Single prompt
python generate.py --checkpoint checkpoints/checkpoint_step7000.pt --prompt "भारत एक"

# Demo mode
python generate.py --checkpoint checkpoints/checkpoint_step7000.pt
```

---

## Technical Details

### Tokenizer
- **Name**: `sarvamai/sarvam-1`
- **Vocab size**: 68,096
- **Type**: BPE (likely)
- **Fertility**: ~1.4 for Hindi (very efficient)
- **Load**: `AutoTokenizer.from_pretrained("sarvamai/sarvam-1")`

### Dataset
- **Name**: `ai4bharat/sangraha`
- **Subset**: `verified/hin` (highest quality Hindi)
- **Total available**: ~12.6B tokens for Hindi
- **We download**: 500M tokens (Chinchilla-optimal for 25M model)
- **Files**: 54 parquet files, ~37 GB total
- **Load**: `load_dataset("ai4bharat/sangraha", data_dir="verified/hin", streaming=True)`

### Chinchilla Scaling Law
- **Rule**: tokens = 20 × parameters
- **25M model**: 500M tokens
- **40M model**: 800M tokens
- **50M model**: 1B tokens

### Memory Requirements (Estimated)
- **Model**: ~100MB (25M params × 4 bytes)
- **Optimizer states**: ~300MB
- **Activations**: ~2GB (batch 64 × seq 1024)
- **Total GPU**: ~4-6GB minimum
- **Recommended**: 8GB+ VRAM

### Training Speed (Estimated)
- **A100**: ~500K tokens/sec
- **RTX 4090**: ~200K tokens/sec
- **RTX 3090**: ~100K tokens/sec
- **500M tokens**: 1-5 hours depending on GPU

---

## File-by-File Reference

### nano_hindi/config.py
```python
# Model config dataclass
GPTConfig(vocab_size, n_layer, n_head, n_kv_head, n_embd, sequence_len, window_pattern)

# Training config dataclass
TrainConfig(batch_size, total_tokens, muon_lr, checkpoint_interval, ...)

# Presets
get_config("25m")  # Returns GPTConfig for 25M model
get_config("40m")
get_config("50m")
```

### nano_hindi/model.py
```python
# Main model class
model = GPT(config)
model.init_weights()  # Call after creation

# Forward pass
loss = model(input_ids, targets)  # Training
logits = model(input_ids)         # Inference

# Generation
for token in model.generate(tokens, max_tokens=100, temperature=0.8):
    print(token)
```

### nano_hindi/muon.py
```python
# Create optimizer
muon = Muon(model.transformer.h.parameters(), lr=0.02, momentum=0.95)

# For distributed training
muon = DistMuon(params, lr=0.02)
```

### train.py
```python
# Command line args
--config: "25m", "40m", or "50m"
--total_tokens: Number of tokens to train on
--checkpoint_dir: Where to save checkpoints
--wandb: Enable wandb logging
--resume: Path to checkpoint to resume from
```

### generate.py
```python
# Command line args
--checkpoint: Path to model checkpoint (required)
--prompt: Text prompt for generation
--interactive: Run in interactive mode
--max_tokens: Maximum tokens to generate
--temperature: Sampling temperature (0 = greedy)
--top_k: Top-k sampling
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in TrainConfig (try 32 or 16)
- Increase `gradient_accumulation_steps` to maintain effective batch size

### Slow Training
- Enable `torch.compile` (default on)
- Use `bfloat16` precision (default)
- Check GPU utilization with `nvidia-smi`

### NaN Loss
- Check data preprocessing (no empty sequences)
- Reduce learning rate
- Enable gradient clipping (default: 1.0)

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

---

## Next Conversation Prompt

Copy this to continue the project:

```
I'm working on nano_hindi, a Hindi language model training project.

Location: d:\Coding_Workspace\nano_hindi

What's done:
- Full codebase created (model, optimizer, data pipeline, training)
- Uses Sarvam tokenizer (68K vocab) + Sangraha Hindi dataset
- 25M param model with modern architecture (RoPE, GQA, ReLU², sliding window)

What I need to do next:
1. Run setup_env.bat to create environment
2. Download data: python data/download.py --max_tokens 500000000
3. Preprocess: python data/preprocess.py
4. Train: python train.py --config 25m

Please read CLAUDE.md for full project details.
```

---

## Links & References

- **Sarvam Tokenizer**: https://huggingface.co/sarvamai/sarvam-1
- **Sangraha Dataset**: https://huggingface.co/datasets/ai4bharat/sangraha
- **Hindi Data Files**: https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/hin
- **Muon Optimizer**: https://kellerjordan.github.io/posts/muon/
- **Chinchilla Paper**: https://arxiv.org/abs/2203.15556
- **modded-nanogpt**: https://github.com/KellerJordan/modded-nanogpt
