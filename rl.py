"""
Reinforcement learning on Hindi GSM8K via simplified GRPO/REINFORCE.

Algorithm (simplified from nanochat's chat_rl.py):
1) No KL regularization (no reference model)
2) On-policy (no PPO ratio+clip)
3) DAPO-style token-level normalization
4) Advantage = reward - mean(group) per question

Single GPU:
    python rl.py --checkpoint sft_checkpoints/sft_step600.pt

2 GPUs:
    torchrun --nproc_per_node=2 rl.py --checkpoint sft_checkpoints/sft_step600.pt --wandb-run hindi-rl-v1

Dry run:
    python rl.py --checkpoint sft_checkpoints/sft_step600.pt --num-iterations 3 --questions-per-step 2 --num-samples 4 --dry-run
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import math
import random
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT
from nano_hindi.muon import Muon, DistMuon
from nano_hindi.tokenizer import get_tokenizer
from nano_hindi.common import setup_distributed, cleanup_distributed, print0

from tasks.hindi_gsm8k import HindiGSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="RL training (REINFORCE/GRPO) for nano_hindi on Hindi GSM8K")
# Logging
parser.add_argument("--wandb-run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
# Runtime
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint")
# RL generation
parser.add_argument("--num-samples", type=int, default=8, help="Number of completions per question")
parser.add_argument("--max-gen-tokens", type=int, default=512, help="Max tokens to generate per completion")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for generation")
parser.add_argument("--top-k", type=int, default=50, help="Top-k for generation sampling (0 = disabled)")
# Training
parser.add_argument("--num-iterations", type=int, default=200, help="Number of RL optimization steps")
parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length for policy forward pass")
parser.add_argument("--device-batch-size", type=int, default=8, help="Per-device batch size for policy forward")
parser.add_argument("--questions-per-step", type=int, default=16, help="Questions sampled per RL step (total across ranks)")
# Optimization
parser.add_argument("--matrix-lr", type=float, default=0.002, help="Muon LR for matrix params")
parser.add_argument("--embedding-lr", type=float, default=0.02, help="AdamW LR for embedding params")
parser.add_argument("--unembedding-lr", type=float, default=0.0004, help="AdamW LR for unembedding params")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--warmup-steps", type=int, default=10, help="LR warmup steps")
# Tool use
parser.add_argument("--use-calculator", action="store_true", help="Enable calculator for <<expr= patterns during generation")
# Evaluation
parser.add_argument("--eval-every", type=int, default=20, help="Evaluate pass@k every N steps")
parser.add_argument("--eval-samples", type=int, default=8, help="Samples per question for pass@k eval")
parser.add_argument("--eval-questions", type=int, default=50, help="Number of val questions to evaluate")
# Output
parser.add_argument("--log-every", type=int, default=1, help="Print training log every N steps")
parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps")
parser.add_argument("--checkpoint-dir", type=str, default="rl_checkpoints", help="Directory for RL checkpoints")
parser.add_argument("--dry-run", action="store_true", help="Skip checkpoint saving")
parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Compute init
ddp, rank, local_rank, world_size = setup_distributed()
master_process = rank == 0
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if torch.cuda.is_available() else nullcontext()
synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if torch.cuda.is_available() else lambda: 0

# Wandb
use_dummy_wandb = args.wandb_run == "dummy" or not master_process
if use_dummy_wandb:
    class DummyWandb:
        def log(self, *a, **kw): pass
        def finish(self, *a, **kw): pass
    wandb_run = DummyWandb()
else:
    import wandb
    wandb_run = wandb.init(project="nano-hindi-rl", name=args.wandb_run, config=vars(args))

# -----------------------------------------------------------------------------
# Load model from SFT checkpoint
print0(f"Loading SFT checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

config_dict = checkpoint["model_config"]
model_config = GPTConfig(**{k: v for k, v in config_dict.items() if k != "head_dim"})
print0(f"Model config: {model_config}")

model = GPT(model_config).to(device)

# Handle torch.compile prefix in state dict
state_dict = checkpoint["model_state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("_orig_mod.", "")
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)
print0(f"Model loaded: {model.num_params():,} parameters")

del checkpoint
if torch.cuda.is_available():
    torch.cuda.empty_cache()

orig_model = model

# Compile
if not args.no_compile and torch.cuda.is_available():
    print0("Compiling model with torch.compile...")
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

# DDP
if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

# Load tokenizer
tokenizer = get_tokenizer()
bos_id = tokenizer.get_bos_token_id()
eos_id = tokenizer.get_eos_token_id()

# -----------------------------------------------------------------------------
# Dataset
print0("Loading Hindi GSM8K dataset...")
train_dataset = HindiGSM8K(split="train")   # ~1055 questions
val_dataset = HindiGSM8K(split="test")       # ~264 questions
print0(f"Train: {len(train_dataset)} questions, Val: {len(val_dataset)} questions")

# Epoch cycling cursor — all ranks advance together but each picks different indices
train_indices = list(range(len(train_dataset)))
random.seed(42)
random.shuffle(train_indices)
cursor = 0
current_epoch = 1

# DDP: split questions across ranks
assert args.questions_per_step % world_size == 0, \
    f"questions_per_step ({args.questions_per_step}) must be divisible by world_size ({world_size})"
questions_per_rank = args.questions_per_step // world_size

def sample_questions(n_per_rank):
    """Sample questions for this rank. All ranks advance cursor by questions_per_step."""
    global cursor, current_epoch
    # Total questions consumed this step (across all ranks)
    total_needed = n_per_rank * world_size
    questions = []
    indices_this_step = []

    for _ in range(total_needed):
        if cursor >= len(train_indices):
            cursor = 0
            current_epoch += 1
            random.shuffle(train_indices)
            print0(f"Starting epoch {current_epoch}")
        indices_this_step.append(train_indices[cursor])
        cursor += 1

    # Each rank picks its slice: rank 0 gets [0, world_size, 2*world_size, ...],
    # rank 1 gets [1, 1+world_size, ...] etc. — interleaved for diversity
    for i in range(rank, total_needed, world_size):
        questions.append(train_dataset[indices_this_step[i]])

    return questions

# -----------------------------------------------------------------------------
# Optimizer setup (same dual Muon+AdamW pattern as sft.py)
def setup_rl_optimizers(model, ddp_mode):
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    model_dim = raw_model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # Matrix params (2D) -> Muon
    matrix_params = list(raw_model.transformer.h.parameters())

    # AdamW groups
    embedding_params = list(raw_model.transformer.wte.parameters())
    lm_head_params = list(raw_model.lm_head.parameters()) if raw_model.lm_head is not None else []
    resid_params = [raw_model.resid_lambdas]
    x0_params = [raw_model.x0_lambdas]
    value_embeds_params = list(raw_model.value_embeds.parameters()) if raw_model.value_embeds else []

    adam_groups = [
        dict(params=embedding_params, lr=args.embedding_lr * dmodel_lr_scale, kind='adamw'),
        dict(params=resid_params, lr=args.matrix_lr * 0.01, kind='adamw'),
        dict(params=x0_params, lr=args.matrix_lr, betas=(0.96, 0.95), kind='adamw'),
    ]
    if lm_head_params:
        adam_groups.insert(0, dict(params=lm_head_params, lr=args.unembedding_lr * dmodel_lr_scale, kind='adamw'))
    if value_embeds_params:
        adam_groups.append(dict(params=value_embeds_params, lr=args.embedding_lr * dmodel_lr_scale, kind='adamw'))

    adamw_opt = torch.optim.AdamW(
        adam_groups,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=args.weight_decay,
        fused=True,
    )

    MuonClass = DistMuon if ddp_mode else Muon
    muon_opt = MuonClass(
        matrix_params,
        lr=args.matrix_lr,
        momentum=0.95,
        weight_decay=args.weight_decay,
    )

    # Store initial LRs
    for opt in [adamw_opt, muon_opt]:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return adamw_opt, muon_opt


adamw_opt, muon_opt = setup_rl_optimizers(orig_model, ddp)
all_params = [p for p in model.parameters() if p.requires_grad]

# DDP no_sync context for gradient accumulation
no_sync_ctx = model.no_sync if ddp else nullcontext

# LR schedule: linear warmup then linear decay to zero
def get_lr_multiplier(step):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    remaining = args.num_iterations - args.warmup_steps
    if remaining <= 0:
        return 1.0
    return max(0.0, 1.0 - (step - args.warmup_steps) / remaining)


# -----------------------------------------------------------------------------
# Calculator for tool use during generation
def simple_calculator(expr):
    """Evaluate a math expression safely. Returns result string or None on failure."""
    try:
        # Restricted namespace: basic math only
        allowed = {"__builtins__": {}, "math": math}
        # Clean up expression
        expr = expr.strip()
        result = eval(expr, allowed)
        # Format: integer if whole number, else float
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# KV cache for fast autoregressive generation
class KVCache:
    """Simple KV cache for autoregressive generation."""

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


# -----------------------------------------------------------------------------
# Batch generation function (with KV cache)
@torch.no_grad()
def generate_batch(
    model_to_use,
    prompt_ids,
    num_samples,
    max_tokens,
    temperature,
    top_k,
    use_calc=False,
    seed=42,
):
    """
    Generate num_samples completions from a single prompt using KV cache.

    Returns:
        sequences: list of full token sequences (prompt + generated)
        masks: list of masks (0 for prompt, 1 for generated tokens)
    """
    raw = model_to_use.module if hasattr(model_to_use, "module") else model_to_use
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    raw.eval()

    prompt_len = len(prompt_ids)
    dev = raw.get_device()
    dev_type = dev.type if hasattr(dev, 'type') else 'cuda'
    rng = torch.Generator(device=dev)
    rng.manual_seed(seed)

    max_total_len = min(prompt_len + max_tokens, args.max_seq_len)

    # Create KV cache
    cache = KVCache(
        batch_size=num_samples,
        max_seq_len=max_total_len,
        n_layers=raw.config.n_layer,
        n_kv_head=raw.config.n_kv_head,
        head_dim=raw.config.head_dim,
        device=dev,
        dtype=ptdtype,
    )

    # Prefill: process entire prompt at once
    prompt_tensor = torch.tensor([prompt_ids] * num_samples, dtype=torch.long, device=dev)
    with torch.autocast(device_type=dev_type, dtype=ptdtype):
        logits = raw(prompt_tensor, kv_cache=cache)  # (B, T, V)
    logits = logits[:, -1, :]  # (B, V) last position

    # Sample first token
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
    if temperature > 0:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1, generator=rng)
    else:
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

    # Track all generated token IDs
    all_ids = [prompt_tensor]  # list of tensors to cat later
    all_ids.append(next_tokens)
    finished = (next_tokens.squeeze(-1) == eos_id)

    # Decode: one token at a time with KV cache
    for gen_step in range(1, max_tokens):
        if finished.all() or cache.get_pos() >= max_total_len:
            break

        # For finished sequences, force BOS (padding token)
        input_token = next_tokens.clone()
        input_token[finished] = bos_id

        with torch.autocast(device_type=dev_type, dtype=ptdtype):
            logits = raw(input_token, kv_cache=cache)  # (B, 1, V)
        logits = logits[:, -1, :]

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

        next_tokens[finished] = bos_id
        all_ids.append(next_tokens)

        newly_finished = (next_tokens.squeeze(-1) == eos_id) & ~finished
        finished = finished | newly_finished

    # Concatenate all token IDs
    ids = torch.cat(all_ids, dim=1)

    # Convert to lists and build masks
    sequences = []
    masks = []
    for i in range(num_samples):
        seq = ids[i].tolist()
        # Trim padding tokens after EOS
        try:
            eos_pos = seq.index(eos_id, prompt_len)
            seq = seq[:eos_pos + 1]  # Include EOS
        except ValueError:
            pass  # No EOS found, keep full sequence
        mask = [0] * prompt_len + [1] * (len(seq) - prompt_len)
        sequences.append(seq)
        masks.append(mask)

    # Optional calculator: post-process generated text to evaluate tool calls
    if use_calc:
        for i in range(num_samples):
            gen_tokens = sequences[i][prompt_len:]
            gen_text = tokenizer.decode(gen_tokens)
            # Find incomplete tool calls: <<expr= without >>
            # Process all <<expr=result>> patterns - check if calculator would give different results
            # For simplicity, we don't re-generate; we just note that the calculator could help
            # The real benefit of calculator would require token-by-token interception
            # For now, just compute results for complete <<expr= patterns
            processed = _process_calculator(gen_text)
            if processed != gen_text:
                new_gen_ids = tokenizer.encode(processed, add_special_tokens=False)
                sequences[i] = sequences[i][:prompt_len] + new_gen_ids
                masks[i] = [0] * prompt_len + [1] * len(new_gen_ids)

    return sequences, masks


def _process_calculator(text):
    """Process <<expr= patterns in generated text, computing results."""
    # Find <<expr= patterns and evaluate them
    def replace_calc(match):
        expr = match.group(1)
        result = simple_calculator(expr)
        if result is not None:
            return f"<<{expr}={result}>>"
        return match.group(0)  # Keep original if eval fails

    # Replace <<expr=anything>> with <<expr=computed_result>>
    processed = re.sub(r'<<([^=]+)=([^>]*)>>', replace_calc, text)
    return processed


# -----------------------------------------------------------------------------
# Advantage computation
def compute_advantages(rewards, num_samples):
    """
    Per-question GRPO-style advantages.
    rewards: flat list, grouped by num_samples per question.
    Returns: flat list of advantages.
    """
    advantages = []
    for i in range(0, len(rewards), num_samples):
        group = rewards[i:i + num_samples]
        mean_r = sum(group) / len(group)
        for r in group:
            advantages.append(r - mean_r)
    return advantages


# -----------------------------------------------------------------------------
# Prepare batch for policy gradient forward pass
def prepare_pg_batch(prompts, completions, advantages_list, max_seq_len):
    """
    Build padded input/target tensors for policy gradient.

    Returns:
        input_ids: (B, T) tensor
        target_ids: (B, T) tensor (shifted, -1 for prompt/padding positions)
        advantages: (B,) tensor
    """
    batch_seqs = []
    prompt_lens = []

    for prompt, completion in zip(prompts, completions):
        full = prompt + completion
        # Truncate to max_seq_len + 1 (need +1 for target shift)
        if len(full) > max_seq_len + 1:
            full = full[:max_seq_len + 1]
        batch_seqs.append(full)
        prompt_lens.append(len(prompt))

    # Pad to same length
    max_len = max(len(s) for s in batch_seqs)
    padded = []
    for seq in batch_seqs:
        padded.append(seq + [bos_id] * (max_len - len(seq)))

    ids = torch.tensor(padded, dtype=torch.long, device=device)
    input_ids = ids[:, :-1]      # (B, T)
    target_ids = ids[:, 1:].clone()  # (B, T)

    # Mask prompt positions and padding
    T = input_ids.size(1)
    for i, pl in enumerate(prompt_lens):
        # Positions 0 to pl-2 in targets correspond to prompt tokens (shift by 1)
        if pl - 1 > 0:
            target_ids[i, :pl - 1] = -1
        # Mask padding: positions beyond actual sequence length
        actual_len = len(batch_seqs[i]) - 1  # -1 because input is shifted
        if actual_len < T:
            target_ids[i, actual_len:] = -1

    adv = torch.tensor(advantages_list, dtype=torch.float, device=device)

    return input_ids, target_ids, adv


# -----------------------------------------------------------------------------
# Pass@k evaluation
@torch.no_grad()
def evaluate_pass_at_k(max_questions=None, num_samples=None, temperature=1.0):
    """Evaluate pass@1 and pass@k on validation set."""
    if max_questions is None:
        max_questions = args.eval_questions
    if num_samples is None:
        num_samples = args.eval_samples

    n_questions = min(max_questions, len(val_dataset))

    # DDP: each rank evaluates a subset
    correct_at_1 = 0
    correct_at_k = 0
    total_reward = 0.0
    total_samples = 0
    num_evaluated = 0

    for idx in range(rank, n_questions, world_size):
        conv = val_dataset[idx]
        prompt_ids = tokenizer.render_for_completion(conv)

        seqs, _ = generate_batch(
            model, prompt_ids, num_samples,
            max_tokens=args.max_gen_tokens,
            temperature=temperature,
            top_k=args.top_k,
            use_calc=False,  # Don't modify tokens
            seed=hash(("eval", idx)) & 0x7FFFFFFF,
        )

        rewards = []
        for seq in seqs:
            gen_tokens = seq[len(prompt_ids):]
            gen_text = tokenizer.decode(gen_tokens)
            # Use calculator for reward evaluation if enabled
            reward_text = _process_calculator(gen_text) if args.use_calculator else gen_text
            r = val_dataset.reward(conv, reward_text)
            rewards.append(r)

        if rewards[0] > 0:
            correct_at_1 += 1
        if any(r > 0 for r in rewards):
            correct_at_k += 1
        total_reward += sum(rewards)
        total_samples += len(rewards)
        num_evaluated += 1

    # Aggregate across ranks
    if ddp:
        stats = torch.tensor([correct_at_1, correct_at_k, total_reward, total_samples, num_evaluated],
                             dtype=torch.float, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        correct_at_1, correct_at_k, total_reward, total_samples, num_evaluated = stats.tolist()

    num_evaluated = max(num_evaluated, 1)
    total_samples = max(total_samples, 1)
    pass_1 = correct_at_1 / num_evaluated
    pass_k = correct_at_k / num_evaluated
    mean_reward = total_reward / total_samples

    return pass_1, pass_k, mean_reward


# -----------------------------------------------------------------------------
# Checkpoint saving
def save_rl_checkpoint(step, pass_at_1=None, mean_reward=None):
    """Save RL checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"rl_step{step}.pt")
    raw_model = orig_model
    ckpt = {
        "step": step,
        "model_state_dict": {k: v.cpu() for k, v in raw_model.state_dict().items()},
        "model_config": model_config.__dict__,
        "rl_config": vars(args),
        "pass_at_1": pass_at_1,
        "mean_reward": mean_reward,
    }
    print0(f"Saving RL checkpoint: {ckpt_path}")
    torch.save(ckpt, ckpt_path)

    # Keep last 3 checkpoints
    import glob as glob_mod
    ckpts = sorted(glob_mod.glob(os.path.join(args.checkpoint_dir, "rl_step*.pt")))
    while len(ckpts) > 3:
        old = ckpts.pop(0)
        os.remove(old)
        print0(f"Removed old checkpoint: {old}")


# -----------------------------------------------------------------------------
# Training loop
print0(f"Starting RL training: {args.num_iterations} steps, {args.questions_per_step} questions/step, {args.num_samples} samples/question")
print0(f"Per-rank: {questions_per_rank} questions/step")
print0(f"LR: matrix={args.matrix_lr}, embedding={args.embedding_lr}")
print0(f"Calculator: {'enabled' if args.use_calculator else 'disabled'}")

total_training_time = 0.0
last_pass_at_1 = None
last_mean_reward = None

for step in range(args.num_iterations):
    # -------------------------------------------------------------------------
    # Evaluation
    if step % args.eval_every == 0:
        print0(f"Step {step}: evaluating pass@k...")
        model.eval()
        with autocast_ctx:
            pass_1, pass_k, eval_mean_reward = evaluate_pass_at_k()
        last_pass_at_1 = pass_1
        print0(f"Step {step} | Pass@1: {pass_1:.4f} | Pass@{args.eval_samples}: {pass_k:.4f} | Mean reward: {eval_mean_reward:.4f}")
        wandb_run.log({
            "step": step,
            "eval/pass_at_1": pass_1,
            f"eval/pass_at_{args.eval_samples}": pass_k,
            "eval/mean_reward": eval_mean_reward,
        })

    # Checkpoint
    if master_process and not args.dry_run and step > 0 and step % args.checkpoint_every == 0:
        save_rl_checkpoint(step, pass_at_1=last_pass_at_1, mean_reward=last_mean_reward)

    # -------------------------------------------------------------------------
    # Rollout phase: generate completions and compute rewards
    synchronize()
    t0 = time.time()

    model.eval()
    all_prompts = []
    all_completions = []
    all_rewards = []

    # Each rank samples its own questions (with different cursor positions via DDP stride)
    rank_questions = sample_questions(questions_per_rank)

    for q_idx, conv in enumerate(rank_questions):
        prompt_ids = tokenizer.render_for_completion(conv)

        with autocast_ctx:
            # Generate WITHOUT calculator (get original model tokens for PG training)
            seqs, masks = generate_batch(
                model, prompt_ids, args.num_samples,
                max_tokens=args.max_gen_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_calc=False,  # Never modify tokens for PG — keep on-policy
                seed=hash((step, rank, q_idx)) & 0x7FFFFFFF,
            )

        for seq, mask in zip(seqs, masks):
            gen_tokens = seq[len(prompt_ids):]
            gen_text = tokenizer.decode(gen_tokens)

            # For reward: optionally use calculator to correct <<expr=result>> answers
            if args.use_calculator:
                reward_text = _process_calculator(gen_text)
            else:
                reward_text = gen_text
            reward = train_dataset.reward(conv, reward_text)

            all_prompts.append(prompt_ids)
            all_completions.append(gen_tokens)  # Original tokens for PG (on-policy)
            all_rewards.append(reward)

    # -------------------------------------------------------------------------
    # Compute advantages (per-question normalization)
    advantages = compute_advantages(all_rewards, args.num_samples)

    # Check if any non-zero advantages exist
    has_signal = any(a != 0.0 for a in advantages)
    mean_reward = sum(all_rewards) / max(len(all_rewards), 1)
    frac_correct = sum(1 for r in all_rewards if r > 0) / max(len(all_rewards), 1)

    if not has_signal:
        # No gradient signal - skip optimizer step
        if step % args.log_every == 0:
            print0(f"Step {step:>4d}/{args.num_iterations} | No gradient signal (all same reward) | "
                   f"Reward: {mean_reward:.3f} | Correct: {frac_correct:.1%}")
        wandb_run.log({
            "step": step,
            "train/mean_reward": mean_reward,
            "train/frac_correct": frac_correct,
            "train/skipped": 1,
        })
        continue

    # -------------------------------------------------------------------------
    # Policy gradient backward pass
    model.train()

    # Zero gradients
    adamw_opt.zero_grad(set_to_none=True)
    muon_opt.zero_grad(set_to_none=True)

    total_pg_loss = 0.0
    total_valid_tokens = 0
    num_sequences = len(all_prompts)

    # Process in mini-batches (use no_sync for all but last to avoid premature DDP all-reduce)
    num_mini_batches = (num_sequences + args.device_batch_size - 1) // args.device_batch_size
    for mb_idx, mb_start in enumerate(range(0, num_sequences, args.device_batch_size)):
        mb_end = min(mb_start + args.device_batch_size, num_sequences)
        is_last_mb = (mb_idx == num_mini_batches - 1)

        mb_input_ids, mb_target_ids, mb_adv = prepare_pg_batch(
            all_prompts[mb_start:mb_end],
            all_completions[mb_start:mb_end],
            advantages[mb_start:mb_end],
            args.max_seq_len,
        )

        sync_ctx = nullcontext() if is_last_mb else no_sync_ctx()

        with sync_ctx:
            with autocast_ctx:
                # Per-token loss: cross_entropy with reduction='none'
                # Returns (B*T,) flattened, with ignore_index=-1 positions set to 0
                per_token_loss = model(mb_input_ids, mb_target_ids, loss_reduction='none')
                per_token_loss = per_token_loss.view(mb_input_ids.size(0), -1)  # (B, T)

                # Mask: only completion tokens contribute
                valid_mask = (mb_target_ids != -1).float()  # (B, T)

                # PG objective: advantage * log_prob = -advantage * ce_loss
                # We want to MINIMIZE the loss, so: loss = sum(ce_loss * advantage) / total_tokens
                # (positive advantage + minimize ce_loss = maximize log_prob for good completions)
                adv_expanded = mb_adv.unsqueeze(1)  # (B, 1)
                pg_loss = (per_token_loss * valid_mask * adv_expanded).sum()
                num_valid = valid_mask.sum()

                total_pg_loss += pg_loss.detach().item()
                total_valid_tokens += num_valid.item()

                # Normalize by total tokens across all mini-batches and sequences
                normalized_loss = pg_loss / num_valid.clamp(min=1)
                normalized_loss.backward()

    # -------------------------------------------------------------------------
    # Optimizer step
    lrm = get_lr_multiplier(step)
    for group in adamw_opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    for group in muon_opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    grad_norm = nn.utils.clip_grad_norm_(all_params, 1.0)

    adamw_opt.step()
    muon_opt.step()
    adamw_opt.zero_grad(set_to_none=True)
    muon_opt.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step > 0:
        total_training_time += dt

    last_mean_reward = mean_reward

    # -------------------------------------------------------------------------
    # Logging
    avg_pg_loss = total_pg_loss / max(total_valid_tokens, 1)

    # Aggregate rewards across ranks for logging
    if ddp:
        reward_tensor = torch.tensor([mean_reward, frac_correct], dtype=torch.float, device=device)
        dist.all_reduce(reward_tensor, op=dist.ReduceOp.AVG)
        mean_reward, frac_correct = reward_tensor.tolist()

    if step % args.log_every == 0:
        print0(
            f"Step {step:>4d}/{args.num_iterations} | "
            f"PG Loss: {avg_pg_loss:.6f} | "
            f"Reward: {mean_reward:.3f} | "
            f"Correct: {frac_correct:.1%} | "
            f"LRM: {lrm:.3f} | "
            f"Grad: {grad_norm:.3f} | "
            f"Epoch: {current_epoch} | "
            f"Step: {dt:.1f}s"
        )

    wandb_run.log({
        "step": step,
        "train/pg_loss": avg_pg_loss,
        "train/mean_reward": mean_reward,
        "train/frac_correct": frac_correct,
        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
        "train/lrm": lrm,
        "train/epoch": current_epoch,
        "train/dt": dt,
    })

# -----------------------------------------------------------------------------
# Final evaluation and checkpoint
print0("Final evaluation...")
model.eval()
with autocast_ctx:
    pass_1, pass_k, eval_mean_reward = evaluate_pass_at_k(
        max_questions=len(val_dataset),  # Full val set
        num_samples=args.eval_samples,
    )
print0(f"Final | Pass@1: {pass_1:.4f} | Pass@{args.eval_samples}: {pass_k:.4f} | Mean reward: {eval_mean_reward:.4f}")
wandb_run.log({
    "step": args.num_iterations,
    "eval/pass_at_1": pass_1,
    f"eval/pass_at_{args.eval_samples}": pass_k,
    "eval/mean_reward": eval_mean_reward,
})

if master_process and not args.dry_run:
    save_rl_checkpoint(args.num_iterations, pass_at_1=pass_1, mean_reward=eval_mean_reward)

# Cleanup
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
wandb_run.finish()
cleanup_distributed()
