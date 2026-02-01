"""
Inference script for nano_hindi.

Usage:
    # Base model (completion mode)
    python generate.py --checkpoint checkpoints/checkpoint_step5000.pt --prompt "भारत एक"
    python generate.py --checkpoint checkpoints/checkpoint_step5000.pt --interactive

    # SFT model (chat mode)
    python generate.py --checkpoint sft_checkpoints/sft_step242.pt --chat
    python generate.py --checkpoint sft_checkpoints/sft_step242.pt --chat --prompt "तुम कौन हो?"
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    config_dict = checkpoint["model_config"]
    config = GPTConfig(**{k: v for k, v in config_dict.items() if k != "head_dim"})

    print(f"Model config: {config}")

    # Create model
    model = GPT(config).to(device)

    # Handle torch.compile prefix (_orig_mod.) in state dict
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        # Strip _orig_mod. prefix if present (from torch.compile)
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # Handle both pretrain checkpoints (have tokens_seen) and SFT checkpoints
    tokens_seen = checkpoint.get('tokens_seen', None)
    if tokens_seen is not None:
        print(f"Loaded from step {checkpoint['step']}, tokens seen: {tokens_seen:,}")
    else:
        print(f"Loaded SFT checkpoint from step {checkpoint['step']}")

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
    """Interactive completion mode (base model)."""
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


def chat_mode(model: GPT, tokenizer, args):
    """
    Interactive chat mode for SFT models.
    Uses the conversation template from nano_hindi/tokenizer.py.
    """
    from nano_hindi.tokenizer import USER_MARKER, ASSISTANT_MARKER

    print("\n" + "=" * 60)
    print("नैनो हिंदी Chat Mode (by Vizuara)")
    print("Type your message and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'clear' to start a new conversation.")
    print("=" * 60 + "\n")

    device = model.get_device()
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # Build conversation context as token IDs
    context_ids = [bos_id]

    while True:
        try:
            user_input = input("आप: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("धन्यवाद! Goodbye!")
                break
            if user_input.lower() == "clear":
                context_ids = [bos_id]
                print("[Conversation cleared]\n")
                continue

            # Encode user turn
            user_text = USER_MARKER + user_input + "\n\n"
            user_ids = tokenizer.encode(user_text, add_special_tokens=False)
            context_ids.extend(user_ids)

            # Encode assistant marker (model generates the rest)
            marker_ids = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
            context_ids.extend(marker_ids)

            # Truncate context if too long (keep last max_seq_len tokens)
            max_ctx = 900  # Leave room for generation
            if len(context_ids) > max_ctx:
                context_ids = [bos_id] + context_ids[-(max_ctx - 1):]

            # Generate
            generated = []
            with torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda', dtype=torch.bfloat16):
                for token in model.generate(
                    context_ids,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed + len(context_ids),  # Vary seed per turn
                ):
                    generated.append(token)
                    if token == eos_id:
                        break

            # Decode only the generated part
            response = tokenizer.decode(generated)
            # Clean up: remove EOS token text and trailing whitespace
            response = response.replace("</s>", "").strip()

            print(f"\nनैनो हिंदी: {response}\n")

            # Add generated tokens to context for multi-turn
            response_ids = tokenizer.encode(response + "\n\n", add_special_tokens=False)
            context_ids.extend(response_ids)

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
        help="Run in interactive completion mode (base model)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run in chat mode (SFT model) with conversation template",
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

    if args.chat:
        chat_mode(model, tokenizer, args)
    elif args.interactive:
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
