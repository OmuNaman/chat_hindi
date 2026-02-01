"""
Hindi instruction-following dataset from AI4Bharat.
https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1

~385K quality-filtered Hindi instructions from:
Dolly, OASST1, HH-RLHF, FLAN v2, WikiHow, Anudesh, LMSys-Chat, NMT-Seed.
Translations filtered by chrF++ back-translation score >= 50.

Each config has different column formats and uses language as the split name ("hi" for Hindi).
"""

from datasets import load_dataset
from tasks.common import Task

# Configs that have a "messages" column (list of {role, content} dicts)
MESSAGE_CONFIGS = ['anudesh', 'oasst1', 'hh-rlhf', 'wikihow', 'lm_sys']

# Configs with instruction/response columns
INSTRUCTION_CONFIGS = ['dolly']

# Configs with inputs/targets columns
FLAN_CONFIGS = ['flan_v2']

# Configs with input_text/output_text columns
NMT_CONFIGS = ['nmt-seed']

ALL_CONFIGS = MESSAGE_CONFIGS + INSTRUCTION_CONFIGS + FLAN_CONFIGS + NMT_CONFIGS


def _extract_messages_from_row(row, config_name):
    """Convert a row from any config format into [{role, content}, ...] messages."""

    if config_name in MESSAGE_CONFIGS:
        # These have a 'messages' column: list of {role, content} dicts
        raw_messages = row["messages"]
        messages = []
        for msg in raw_messages:
            role = msg.get("role", "")
            # lm_sys has 'backtranslated_content' as the Hindi version
            if config_name == "lm_sys":
                content = msg.get("backtranslated_content") or msg.get("content", "")
            else:
                content = msg.get("content", "")
            if role in ("user", "human"):
                messages.append({"role": "user", "content": content})
            elif role in ("assistant", "bot", "chatbot"):
                messages.append({"role": "assistant", "content": content})
        return messages

    elif config_name == "dolly":
        instruction = row.get("instruction", "") or ""
        context = row.get("context", "") or ""
        response = row.get("response", "") or ""
        user_content = f"{instruction}\n\n{context}".strip() if context.strip() else instruction.strip()
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response.strip()},
        ]

    elif config_name == "flan_v2":
        inputs = row.get("inputs", "") or ""
        targets = row.get("targets", "") or ""
        return [
            {"role": "user", "content": inputs.strip()},
            {"role": "assistant", "content": targets.strip()},
        ]

    elif config_name == "nmt-seed":
        input_text = row.get("input_text", "") or ""
        output_text = row.get("output_text", "") or ""
        return [
            {"role": "user", "content": input_text.strip()},
            {"role": "assistant", "content": output_text.strip()},
        ]

    return []


class HindiInstruct(Task):
    """
    ai4bharat/indic-instruct-data-v0.1 dataset.
    Loads all configs (Hindi split) and converts to user/assistant conversations.
    """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "HindiInstruct split must be train|test"
        self.split = split

        # Load all configs, extracting Hindi split
        self.examples = []
        for config in ALL_CONFIGS:
            try:
                # Most configs use "hi" as the split for Hindi data
                ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", config, split="hi")
                count = 0
                for row in ds:
                    messages = _extract_messages_from_row(row, config)
                    # Filter: need at least 1 user + 1 assistant message, non-empty
                    if (len(messages) >= 2
                            and any(m["role"] == "user" and m["content"] for m in messages)
                            and any(m["role"] == "assistant" and m["content"] for m in messages)):
                        self.examples.append(messages)
                        count += 1
                print(f"  Loaded {config} (hi): {count} conversations")
            except Exception:
                # Some configs might not have a "hi" split, try "train" as fallback
                try:
                    ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", config, split="train")
                    count = 0
                    for row in ds:
                        messages = _extract_messages_from_row(row, config)
                        if (len(messages) >= 2
                                and any(m["role"] == "user" and m["content"] for m in messages)
                                and any(m["role"] == "assistant" and m["content"] for m in messages)):
                            self.examples.append(messages)
                            count += 1
                    print(f"  Loaded {config} (train): {count} conversations")
                except Exception as e2:
                    print(f"  Warning: skipping config '{config}': {e2}")

        # Deterministic shuffle
        import random
        rng = random.Random(42)
        rng.shuffle(self.examples)

        # 95/5 train/test split
        total = len(self.examples)
        split_idx = int(total * 0.95)

        if split == "train":
            self.examples = self.examples[:split_idx]
        else:
            self.examples = self.examples[split_idx:]

        self.length = len(self.examples)
        print(f"  HindiInstruct {split}: {self.length} conversations")

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.examples[index]
        # Ensure alternating user/assistant, truncate to valid pairs
        valid = []
        for i, msg in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg["role"] == expected_role:
                valid.append(msg)
            else:
                break
        # Must end on assistant
        if len(valid) % 2 != 0:
            valid = valid[:-1]
        if len(valid) < 2:
            # Fallback: just use first user/assistant pair we can find
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
            if user_msg and asst_msg:
                valid = [user_msg, asst_msg]
            else:
                valid = [{"role": "user", "content": "नमस्ते"}, {"role": "assistant", "content": "नमस्ते!"}]
        return {"messages": valid}


if __name__ == "__main__":
    print("Loading HindiInstruct train split...")
    task = HindiInstruct(split="train")
    print(f"Train size: {len(task)}")

    print("\nFirst 3 examples:")
    for i in range(3):
        ex = task[i]
        print("=" * 80)
        for msg in ex['messages']:
            print(f"  [{msg['role']}]: {msg['content'][:200]}...")
