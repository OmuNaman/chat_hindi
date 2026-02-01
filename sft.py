"""
Supervised fine-tuning (SFT) for nano_hindi.

Finetunes the pretrained 250M Hindi model on conversational data.
Closely follows nanochat's chat_sft.py architecture.

Single GPU:
    python sft.py --checkpoint checkpoints/checkpoint_step4768.pt

Multi-GPU (2×H100):
    torchrun --nproc_per_node=2 sft.py --checkpoint checkpoints/checkpoint_step4768.pt --run hindi-sft-v1

Dry run (no checkpoints):
    python sft.py --checkpoint checkpoints/checkpoint_step4768.pt --num-iterations 5 --device-batch-size 4 --dry-run
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT
from nano_hindi.muon import Muon, DistMuon
from nano_hindi.eval import evaluate_bpb
from nano_hindi.tokenizer import get_tokenizer, compute_token_bytes
from nano_hindi.common import setup_distributed, cleanup_distributed, print0

from tasks.common import TaskMixture
from tasks.hindi_instruct import HindiInstruct
from tasks.hindi_mmlu import HindiMMLU
from tasks.hindi_gsm8k import HindiGSM8K
from tasks.hindi_spelling import HindiSpelling, HindiLetterCount
from tasks.customjson import CustomJSON

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) for nano_hindi")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
# Runtime
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="Number of optimization steps (-1 = full epoch)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=1024, help="Max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="Total batch size in tokens")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="LR for embedding parameters (AdamW)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="LR for unembedding parameters (AdamW)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW")
parser.add_argument("--init-lr-frac", type=float, default=0.1, help="Initial LR as fraction of base LR")
parser.add_argument("--warmup-steps", type=int, default=30, help="LR warmup steps (linear from init-lr-frac to 1.0)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="Evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=10_485_760, help="Tokens to evaluate val loss on (~10M)")
# Logging & inference
parser.add_argument("--log-every", type=int, default=10, help="Print training log every N steps")
parser.add_argument("--inference-every", type=int, default=50, help="Generate inference samples every N steps (-1 = disable)")
parser.add_argument("--checkpoint-every", type=int, default=200, help="Save checkpoint every N steps (-1 = only at end)")
# Output
parser.add_argument("--checkpoint-dir", type=str, default="sft_checkpoints", help="Directory for SFT checkpoints")
parser.add_argument("--dry-run", action="store_true", help="Log to wandb but skip checkpoints")
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
use_dummy_wandb = args.run == "dummy" or not master_process
if use_dummy_wandb:
    class DummyWandb:
        def log(self, *a, **kw): pass
        def finish(self, *a, **kw): pass
    wandb_run = DummyWandb()
else:
    import wandb
    wandb_run = wandb.init(project="nano-hindi-sft", name=args.run, config=vars(args))

# -----------------------------------------------------------------------------
# Load model from pretrained checkpoint
print0(f"Loading pretrained checkpoint: {args.checkpoint}")
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

del checkpoint  # Free memory
torch.cuda.empty_cache()

orig_model = model

# Compile
if not args.no_compile and torch.cuda.is_available():
    print0("Compiling model with torch.compile...")
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

# DDP
if ddp:
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

depth = model_config.n_layer

# Batch size math
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, \
    f"total_batch_size ({args.total_batch_size}) must be divisible by " \
    f"device_batch_size * max_seq_len * world_size ({world_tokens_per_fwdbwd})"
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Load tokenizer
tokenizer = get_tokenizer()
token_bytes = compute_token_bytes(device=device)

# -----------------------------------------------------------------------------
# Setup optimizer (same pattern as train.py but in sft.py, no modification to existing files)
def setup_sft_optimizers(model, ddp_mode):
    """Setup Muon + AdamW optimizers for SFT."""
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

    # Scale scalar LRs from matrix_lr (not hardcoded pretraining values)
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

    # Store initial LRs (warmup schedule handles init_lr_frac)
    for opt in [adamw_opt, muon_opt]:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return adamw_opt, muon_opt


adamw_opt, muon_opt = setup_sft_optimizers(orig_model, ddp)

# Cache all params for grad clipping
all_params = [p for p in model.parameters() if p.requires_grad]

# DDP no_sync context
no_sync_ctx = model.no_sync if ddp else nullcontext

# -----------------------------------------------------------------------------
# SFT data mixture
print0("Loading SFT datasets...")
identity_path = os.path.join(os.path.dirname(__file__), "identity_conversations.jsonl")

train_dataset = TaskMixture([
    HindiInstruct(split="train"),                    # ~365K rows
    HindiMMLU(split="train"),                        # ~85K rows (knowledge MCQ)
    HindiGSM8K(split="train"),                       # ~1055 rows (Hindi math)
    HindiGSM8K(split="train"),                       # 2 epochs of GSM8K (small dataset)
    HindiSpelling(size=50000, split="train"),         # 50K rows (spell Hindi words)
    HindiLetterCount(size=20000, split="train"),      # 20K rows (count characters)
    CustomJSON(filepath=identity_path),               # ~40 identity conversations
    CustomJSON(filepath=identity_path),               # 2 epochs of identity
])
val_dataset = TaskMixture([
    HindiInstruct(split="test"),
    HindiMMLU(split="test"),
    HindiGSM8K(split="test"),
])
print0(f"Train dataset: {len(train_dataset):,} conversations")
print0(f"Val dataset: {len(val_dataset):,} conversations")

# Estimate total steps
if args.num_iterations > 0:
    total_steps = args.num_iterations
else:
    # One full epoch: each conversation consumed once across all ranks
    # Each step consumes device_batch_size * grad_accum_steps * world_size conversations (approx)
    total_steps = len(train_dataset) // (args.device_batch_size * grad_accum_steps * world_size)
print0(f"Total steps: {total_steps:,} ({'user-set' if args.num_iterations > 0 else 'estimated 1 epoch'})")
print0(f"Warmup steps: {args.warmup_steps}")

# -----------------------------------------------------------------------------
# DataLoader: BOS-aligned bestfit packing (ported from nanochat)

last_step = False
approx_progress = 0.0
current_epoch = 1


def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row starts with BOS. Conversations are packed using best-fit algorithm.
    When no conversation fits, the row is padded (never cropped).
    Padding positions have targets masked with -1 (ignore_index).
    Additionally, non-assistant tokens are masked with -1 (only predict assistant).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position

    # Conversation buffer: list of (token_ids, mask) tuples
    conv_buffer = []
    cursor = rank if ddp else 0
    consumed = cursor
    epoch = 1
    it = 0
    step_size = world_size if ddp else 1
    skipped_long = 0
    skipped_no_label = 0

    def refill_buffer():
        nonlocal cursor, epoch, skipped_long, skipped_no_label
        max_attempts = buffer_size * 50
        attempts = 0
        while len(conv_buffer) < buffer_size and attempts < max_attempts:
            conversation = dataset[cursor % dataset_size]
            ids, mask = tokenizer.render_conversation(conversation)
            cursor += step_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
            attempts += 1
            # Skip conversations that can never fit in a row
            if len(ids) > row_capacity:
                skipped_long += 1
                continue
            # Skip conversations with zero assistant tokens (nothing to learn)
            if not any(m == 1 for m in mask):
                skipped_no_label += 1
                continue
            conv_buffer.append((ids, mask))
        if len(conv_buffer) == 0 and attempts >= max_attempts:
            raise RuntimeError(
                f"No valid conversations found after {attempts} attempts. "
                f"Skipped {skipped_long} too-long (>{row_capacity} tokens) "
                f"and {skipped_no_label} with zero assistant tokens."
            )

    while True:
        rows = []
        row_masks = []
        row_lengths = []

        for _ in range(args.device_batch_size):
            row = []
            row_mask = []

            while len(row) < row_capacity:
                if len(conv_buffer) < buffer_size:
                    refill_buffer()
                if not conv_buffer:
                    break

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (ids, mask) in enumerate(conv_buffer):
                    conv_len = len(ids)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    ids, mask = conv_buffer.pop(best_idx)
                    row.extend(ids)
                    row_mask.extend(mask)
                    consumed += step_size
                elif len(row) == 0:
                    # Row is empty but nothing fits - buffer is stale, flush and retry
                    conv_buffer.clear()
                    refill_buffer()
                    continue
                else:
                    break  # No conversation fits remaining space

            # Record content length, then pad any remaining space
            content_len = len(row)
            if content_len < row_capacity:
                pad_len = row_capacity - content_len
                bos_id = tokenizer.get_bos_token_id()
                row.extend([bos_id] * pad_len)
                row_mask.extend([0] * pad_len)

            row_lengths.append(content_len)
            rows.append(row[:row_capacity])
            row_masks.append(row_mask[:row_capacity])

        # Safety: skip batches with zero valid targets (all padding)
        has_valid = any(
            any(m == 1 for m in row_masks[i][1:])
            for i in range(len(rows))
        )
        if not has_valid:
            # Log once
            if skipped_long > 0 and it == 0:
                print0(f"Warning: skipped {skipped_long} conversations longer than {row_capacity} tokens")
            continue

        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        # Build tensors
        use_cuda = torch.cuda.is_available()
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        mask_tensor = torch.tensor(row_masks, dtype=torch.long, pin_memory=use_cuda)

        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.long, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        target_mask = mask_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        # Mask targets: set non-predicted positions to -1 (ignore_index)
        targets = torch.where(target_mask == 1, targets, torch.full_like(targets, -1))

        # Also mask padding positions
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1

        if it == 1 and (skipped_long > 0 or skipped_no_label > 0):
            print0(f"Packing stats: {skipped_long} too-long, {skipped_no_label} no-label conversations skipped")

        yield inputs, targets


train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0.0


# LR schedule: warmup, then constant for 80%, then linear decay to 0
def get_lr_multiplier(step, progress):
    # Phase 1: Linear warmup from init_lr_frac to 1.0
    if args.warmup_steps > 0 and step < args.warmup_steps:
        warmup_frac = step / args.warmup_steps
        return args.init_lr_frac + (1.0 - args.init_lr_frac) * warmup_frac
    # Phase 2: Constant LR for 80% of training
    if progress < 0.8:
        return 1.0
    # Phase 3: Linear decay to 0
    return 1.0 - (progress - 0.8) / 0.2


# Muon momentum warmup
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


# -----------------------------------------------------------------------------
# Chat inference for monitoring during SFT
SFT_INFERENCE_PROMPTS = [
    "भारत की राजधानी क्या है?",
    "तुम कौन हो?",
    "मुझे पानी पूरी की रेसिपी बताओ।",
]

def run_sft_inference(model, tokenizer_obj, prompts, max_tokens=100, temperature=0.8):
    """Generate chat-mode samples for monitoring SFT training."""
    from nano_hindi.tokenizer import USER_MARKER, ASSISTANT_MARKER

    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    raw_model.eval()

    bos_id = tokenizer_obj.get_bos_token_id()
    eos_id = tokenizer_obj.get_eos_token_id()
    results = []

    for prompt in prompts:
        # Build chat context: BOS + user turn + assistant marker
        context_text = USER_MARKER + prompt + "\n\n"
        context_ids = [bos_id] + tokenizer_obj.encode(context_text, add_special_tokens=False)
        marker_ids = tokenizer_obj.encode(ASSISTANT_MARKER, add_special_tokens=False)
        context_ids.extend(marker_ids)

        generated = []
        with torch.autocast(device_type="cuda", dtype=ptdtype):
            for token in raw_model.generate(
                context_ids, max_tokens=max_tokens, temperature=temperature, top_k=50
            ):
                generated.append(token)
                if token == eos_id:
                    break

        response = tokenizer_obj.decode(generated).replace("</s>", "").strip()
        results.append((prompt, response))

    raw_model.train()
    return results


# -----------------------------------------------------------------------------
# Checkpoint saving
def save_sft_checkpoint(model, step, val_bpb=None):
    """Save SFT checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"sft_step{step}.pt")
    raw_model = orig_model
    ckpt = {
        "step": step,
        "model_state_dict": {k: v.cpu() for k, v in raw_model.state_dict().items()},
        "model_config": model_config.__dict__,
        "sft_config": vars(args),
        "val_bpb": val_bpb,
    }
    print0(f"Saving SFT checkpoint: {ckpt_path}")
    torch.save(ckpt, ckpt_path)

    # Keep last 3 checkpoints
    import glob as glob_mod
    ckpts = sorted(glob_mod.glob(os.path.join(args.checkpoint_dir, "sft_step*.pt")))
    while len(ckpts) > 3:
        old = ckpts.pop(0)
        os.remove(old)
        print0(f"Removed old checkpoint: {old}")


# -----------------------------------------------------------------------------
# Training loop
print0("Starting SFT training...")
x, y = next(train_loader)
min_val_bpb = float("inf")
val_bpb = None
smooth_train_loss = 0.0
ema_beta = 0.9
total_training_time = 0.0
step = 0

while True:
    # Sync last_step across ranks
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # Evaluation
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * world_size)
        eval_steps = max(eval_steps, 1)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model if not ddp else model.module,
                                   val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "total_training_time": total_training_time, "val/bpb": val_bpb})
        model.train()

    # Periodic checkpoint saving
    if master_process and not args.dry_run:
        should_save = last_step
        if not should_save and args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0:
            should_save = True
        if should_save:
            save_sft_checkpoint(model, step, val_bpb=val_bpb)

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Training step
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        is_last_micro = (micro_step == grad_accum_steps - 1)
        sync_ctx = nullcontext() if is_last_micro else no_sync_ctx()

        # Pre-forward guard: skip micro-batches with zero valid targets
        if (y != -1).sum().item() == 0:
            print0(f"Skipping empty-target batch at step {step}, micro_step {micro_step}")
            x, y = next(train_loader)
            continue

        with sync_ctx:
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()

            # NaN detection
            if not torch.isfinite(train_loss):
                print0(f"NaN/Inf loss at step {step}, micro_step {micro_step}")
                valid_targets = (y != -1).sum().item()
                print0(f"  Valid targets: {valid_targets} / {y.numel()}")
                print0(f"  Input range: [{x.min().item()}, {x.max().item()}]")
                break

            loss = loss / grad_accum_steps
            loss.backward()

        x, y = next(train_loader)
        progress = max(progress, approx_progress)
    else:
        # Only run optimizer if we didn't break out of the loop
        # LR schedule
        lrm = get_lr_multiplier(step, progress)
        muon_momentum = get_muon_momentum(step)
        for group in adamw_opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        for group in muon_opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            group["momentum"] = muon_momentum

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(all_params, 1.0)

        # Check for NaN gradients
        if not torch.isfinite(grad_norm):
            print0(f"NaN/Inf gradient norm at step {step}")

        # Optimizer step
        adamw_opt.step()
        muon_opt.step()
        adamw_opt.zero_grad(set_to_none=True)
        muon_opt.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    step += 1

    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    if step > 10:
        total_training_time += dt

    if step % args.log_every == 0 or step == 1:
        print0(
            f"Step {step:>6d}/{total_steps} ({pct_done:.1f}%) | "
            f"Loss {debiased_loss:.4f} | LRM {lrm:.3f} | "
            f"Speed {tok_per_sec/1e3:.0f}K tok/s | "
            f"Step {dt:.2f}s | Epoch {current_epoch} | "
            f"Time {total_training_time / 60:.1f}m"
        )

    if step % args.log_every == 0:
        wandb_run.log({
            "step": step,
            "total_training_time": total_training_time,
            "train/loss": debiased_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/epoch": current_epoch,
        })

    # Inference samples
    if master_process and args.inference_every > 0 and step > 0 and step % args.inference_every == 0:
        print0("\n--- SFT Inference Samples ---")
        samples = run_sft_inference(model, tokenizer, SFT_INFERENCE_PROMPTS)
        for prompt, response in samples:
            print0(f"  Q: {prompt}")
            print0(f"  A: {response[:300]}")
            print0("")
        print0("-----------------------------\n")
        model.train()
        if ddp:
            dist.barrier()

# Print final stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")
print0(f"Total steps: {step}")

wandb_run.finish()
cleanup_distributed()
