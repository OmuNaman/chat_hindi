"""
Interactive QA demo for fine-tuned nano_hindi model.

Usage:
    python qa_demo.py --checkpoint checkpoints/qa/best.pt
    python qa_demo.py --checkpoint checkpoints/qa/best.pt --temperature 0.3
"""

import argparse

import torch
from transformers import AutoTokenizer

from nano_hindi.config import GPTConfig
from nano_hindi.model import GPT
from finetune.config import FinetuneConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned model."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config
    model_config_dict = checkpoint["model_config"]
    model_config = GPTConfig(**{k: v for k, v in model_config_dict.items() if k != "head_dim"})

    # Get finetune config
    finetune_config_dict = checkpoint.get("finetune_config", {})
    finetune_config = FinetuneConfig(**{k: v for k, v in finetune_config_dict.items() if k in FinetuneConfig.__dataclass_fields__})

    # Create and load model
    model = GPT(model_config).to(device)

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # Print metrics from training
    metrics = checkpoint.get("metrics", {})
    if metrics:
        print(f"  Training F1: {metrics.get('f1', 'N/A'):.4f}")
        print(f"  Training EM: {metrics.get('em', 'N/A'):.4f}")

    return model, finetune_config


@torch.no_grad()
def answer_question(
    model,
    tokenizer,
    context: str,
    question: str,
    config: FinetuneConfig,
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> str:
    """Generate answer for a question given context."""
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
        print(f"(Context truncated from {len(input_ids)} to {max_input_len} tokens)")
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


def interactive_mode(model, tokenizer, config: FinetuneConfig, args):
    """Interactive QA mode."""
    print("\n" + "=" * 60)
    print("nano_hindi Question Answering Demo")
    print("=" * 60)
    print("\nHow to use:")
    print("1. Enter a context paragraph (in Hindi)")
    print("2. Ask a question about the context")
    print("3. Get the model's answer")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the demo")
    print("  'demo' - Run demo examples")
    print("  'clear' - Clear context")
    print("=" * 60)

    current_context = None

    while True:
        try:
            # Get context
            if current_context is None:
                print("\n[Enter context paragraph (or 'demo' for examples)]")
                context = input("संदर्भ: ").strip()

                if context.lower() in ["quit", "exit", "q"]:
                    print("धन्यवाद! Goodbye!")
                    break

                if context.lower() == "demo":
                    run_demo(model, tokenizer, config, args)
                    continue

                if not context:
                    print("Please enter a context.")
                    continue

                current_context = context
                print(f"\n[Context set: {len(context)} chars]")
            else:
                print(f"\n[Current context: {current_context[:50]}...]")

            # Get question
            print("\n[Enter question (or 'clear' to change context)]")
            question = input("प्रश्न: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("धन्यवाद! Goodbye!")
                break

            if question.lower() == "clear":
                current_context = None
                print("[Context cleared]")
                continue

            if not question:
                print("Please enter a question.")
                continue

            # Generate answer
            print("\n[Generating answer...]")
            answer = answer_question(
                model, tokenizer,
                current_context, question,
                config,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            print(f"\nउत्तर: {answer}")

        except KeyboardInterrupt:
            print("\n\nधन्यवाद! Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def run_demo(model, tokenizer, config: FinetuneConfig, args):
    """Run demo examples."""
    demos = [
        {
            "context": "महात्मा गांधी का जन्म 2 अक्टूबर 1869 को पोरबंदर में हुआ था। उनका पूरा नाम मोहनदास करमचंद गांधी था। वे भारत के राष्ट्रपिता माने जाते हैं।",
            "questions": [
                "गांधी जी का जन्म कब हुआ था?",
                "गांधी जी का पूरा नाम क्या था?",
                "गांधी जी का जन्म कहाँ हुआ था?",
            ]
        },
        {
            "context": "भारत की राजधानी नई दिल्ली है। यह यमुना नदी के किनारे बसा हुआ है। दिल्ली में लाल किला, कुतुब मीनार और इंडिया गेट जैसे प्रसिद्ध स्थल हैं।",
            "questions": [
                "भारत की राजधानी क्या है?",
                "दिल्ली किस नदी के किनारे है?",
                "दिल्ली में कौन से प्रसिद्ध स्थल हैं?",
            ]
        },
        {
            "context": "हिंदी भारत की राजभाषा है। यह दुनिया की चौथी सबसे ज्यादा बोली जाने वाली भाषा है। हिंदी दिवस हर साल 14 सितंबर को मनाया जाता है।",
            "questions": [
                "हिंदी दिवस कब मनाया जाता है?",
                "हिंदी कौन सी सबसे ज्यादा बोली जाने वाली भाषा है?",
            ]
        },
    ]

    print("\n" + "=" * 60)
    print("DEMO EXAMPLES")
    print("=" * 60)

    for i, demo in enumerate(demos):
        print(f"\n--- Demo {i + 1} ---")
        print(f"संदर्भ: {demo['context'][:100]}...")

        for question in demo['questions']:
            print(f"\nप्रश्न: {question}")

            answer = answer_question(
                model, tokenizer,
                demo['context'], question,
                config,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            print(f"उत्तर: {answer}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Interactive QA demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/qa/best.pt",
        help="Path to fine-tuned checkpoint",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo examples and exit",
    )

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    if args.demo:
        run_demo(model, tokenizer, config, args)
    else:
        interactive_mode(model, tokenizer, config, args)


if __name__ == "__main__":
    main()
