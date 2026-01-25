"""
Inference script for nano_hindi.

Usage:
    python generate.py --checkpoint checkpoints/checkpoint_step5000.pt --prompt "भारत एक"
    python generate.py --checkpoint checkpoints/checkpoint_step5000.pt --interactive
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config
    config_dict = checkpoint["model_config"]
    config = GPTConfig(**{k: v for k, v in config_dict.items() if k != "head_dim"})

    print(f"Model config: {config}")

    # Create and load model
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded from step {checkpoint['step']}, tokens seen: {checkpoint['tokens_seen']:,}")

    return model, config


def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    seed: int = 42,
):
    """Generate text from a prompt."""
    device = model.get_device()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt tokens: {len(input_ids)}")

    # Generate
    generated = []
    for token in model.generate(
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    ):
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break

    # Decode
    output = tokenizer.decode(input_ids + generated)
    return output


def interactive_mode(model: GPT, tokenizer, args):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("nano_hindi Interactive Mode")
    print("Type your Hindi prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("आप: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["quit", "exit", "q"]:
                print("धन्यवाद! Goodbye!")
                break

            output = generate_text(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
            )

            print(f"\nमॉडल: {output}\n")

        except KeyboardInterrupt:
            print("\nधन्यवाद! Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text with nano_hindi")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        output = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {output}")
    else:
        # Demo with some Hindi prompts
        demo_prompts = [
            "भारत एक",
            "आज का मौसम",
            "हिंदी भाषा बहुत",
            "मेरा नाम",
        ]

        print("\n--- Demo Generation ---\n")
        for prompt in demo_prompts:
            output = generate_text(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
            print("-" * 40)


if __name__ == "__main__":
    main()
