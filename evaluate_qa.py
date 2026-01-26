"""
Evaluation script for fine-tuned nano_hindi QA model.

Computes F1 and Exact Match (EM) scores on the validation set.

Usage:
    python evaluate_qa.py --checkpoint checkpoints/qa/best.pt
    python evaluate_qa.py --checkpoint checkpoints/qa/best.pt --output results.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT
from finetune.config import FinetuneConfig
from finetune.dataset import load_indicqa, QAExample


def load_finetuned_model(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get configs
    model_config_dict = checkpoint["model_config"]
    model_config = GPTConfig(**{k: v for k, v in model_config_dict.items() if k != "head_dim"})

    finetune_config_dict = checkpoint.get("finetune_config", {})
    finetune_config = FinetuneConfig(**{k: v for k, v in finetune_config_dict.items() if k in FinetuneConfig.__dataclass_fields__})

    # Create model
    model = GPT(model_config).to(device)

    # Load state dict
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # Print info
    metrics = checkpoint.get("metrics", {})
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Training F1: {metrics.get('f1', 'N/A')}")
    print(f"  Training EM: {metrics.get('em', 'N/A')}")

    return model, model_config, finetune_config


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    text = text.strip().lower()
    # Remove Hindi punctuation
    text = re.sub(r'[।॥,\.\?!:;\'\"\(\)\[\]{}]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_exact_match(pred: str, gold: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(pred) == normalize_answer(gold))


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    context: str,
    question: str,
    config: FinetuneConfig,
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> str:
    """Generate answer for a given context and question."""
    device = model.get_device()

    # Format input
    input_text = (
        f"{config.context_prefix}{context}"
        f"{config.question_prefix}{question}"
        f"{config.answer_prefix}"
    )

    # Tokenize
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)

    # Truncate if needed
    max_input_len = 512 - max_tokens
    if len(input_ids) > max_input_len:
        input_ids = input_ids[:max_input_len]

    # Generate
    generated = []
    for token in model.generate(
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=1 if temperature == 0 else 50,
    ):
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)

    # Decode
    answer = tokenizer.decode(generated).strip()
    return answer


def evaluate(args):
    """Main evaluation function."""
    device = args.device

    # Load model
    model, model_config, finetune_config = load_finetuned_model(args.checkpoint, device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_config.tokenizer_name)

    # Load data
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = finetune_config.data_path

    print(f"\nLoading data from {data_path}")
    examples = load_indicqa(data_path)

    # Use validation split
    if args.split == "val":
        split_idx = int(len(examples) * finetune_config.train_split)
        examples = examples[split_idx:]
    elif args.split == "train":
        split_idx = int(len(examples) * finetune_config.train_split)
        examples = examples[:split_idx]

    print(f"Evaluating on {len(examples)} examples")

    # Filter out unanswerable questions (same as training)
    if finetune_config.filter_unanswerable:
        examples = [e for e in examples if e.category != "NO" and e.answer]
        print(f"After filtering unanswerable: {len(examples)} examples")

    # Limit examples if specified
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Limited to {len(examples)} examples")

    # Evaluate
    results = []
    f1_scores = []
    em_scores = []

    # Category-wise metrics
    category_metrics = defaultdict(lambda: {"f1": [], "em": []})

    for example in tqdm(examples, desc="Evaluating"):
        # Generate prediction
        pred_answer = generate_answer(
            model, tokenizer,
            example.context, example.question,
            finetune_config,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Get gold answer (with filter_unanswerable=True, all examples have answers)
        gold_answer = example.answer if example.answer else finetune_config.no_answer_text.strip()

        # Compute metrics
        f1 = compute_f1(pred_answer, gold_answer)
        em = compute_exact_match(pred_answer, gold_answer)

        f1_scores.append(f1)
        em_scores.append(em)

        # Category metrics
        category_metrics[example.category]["f1"].append(f1)
        category_metrics[example.category]["em"].append(em)

        # Store result
        results.append({
            "context": example.context[:200] + "..." if len(example.context) > 200 else example.context,
            "question": example.question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "category": example.category,
            "f1": f1,
            "em": em,
        })

    # Compute overall metrics
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall:")
    print(f"  F1 Score:    {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print(f"  Exact Match: {avg_em:.4f} ({avg_em*100:.2f}%)")
    print(f"  Samples:     {len(examples)}")

    print(f"\nBy Category:")
    for category, metrics in category_metrics.items():
        cat_f1 = sum(metrics["f1"]) / len(metrics["f1"])
        cat_em = sum(metrics["em"]) / len(metrics["em"])
        print(f"  {category}:")
        print(f"    F1:  {cat_f1:.4f} ({len(metrics['f1'])} samples)")
        print(f"    EM:  {cat_em:.4f}")

    # Show some examples
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")

    for i, r in enumerate(results[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {r['question']}")
        print(f"Gold:     {r['gold_answer']}")
        print(f"Pred:     {r['pred_answer']}")
        print(f"F1: {r['f1']:.3f}, EM: {r['em']:.0f}")

    # Save results
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "num_examples": len(examples),
            "metrics": {
                "f1": avg_f1,
                "em": avg_em,
            },
            "category_metrics": {
                cat: {
                    "f1": sum(m["f1"]) / len(m["f1"]),
                    "em": sum(m["em"]) / len(m["em"]),
                    "count": len(m["f1"]),
                }
                for cat, m in category_metrics.items()
            },
            "predictions": results,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {args.output}")

    return avg_f1, avg_em


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned QA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/qa/best.pt",
        help="Path to fine-tuned checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to evaluation data (default: use config)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "all"],
        help="Data split to evaluate",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum examples to evaluate",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0 for greedy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
