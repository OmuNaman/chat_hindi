"""
Hindi GSM8K - Grade school math in Hindi.
https://huggingface.co/datasets/nvidia/GSM8K-Hi

1,319 math word problems translated from English GSM8K by NVIDIA.
Uses <<expression=result>> tool call format for calculator steps.

Since there's only a test split, we split 80/20 for train/test
and oversample (2 epochs) during training since it's small.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """Extract the numerical answer after #### marker."""
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return match_str
    return None


class HindiGSM8K(Task):
    """
    nvidia/GSM8K-Hi dataset.
    Hindi math word problems with step-by-step solutions.
    """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "HindiGSM8K split must be train|test"
        self.split = split

        # Only has 'test' split, so we split it ourselves
        ds = load_dataset("nvidia/GSM8K-Hi", split="test").shuffle(seed=42)
        total = len(ds)
        split_idx = int(total * 0.8)  # ~1055 train, ~264 test

        if split == "train":
            self.ds = ds.select(range(split_idx))
        else:
            self.ds = ds.select(range(split_idx, total))

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]

        # Parse <<expr=result>> tool calls into parts (same format as nanochat GSM8K)
        assistant_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                assistant_parts.append({"type": "python", "text": expr})
                assistant_parts.append({"type": "python_output", "text": result})
            else:
                assistant_parts.append({"type": "text", "text": part})

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_parts},
        ]

        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """Compare numerical answer after #### marker."""
        assert isinstance(assistant_response, str)
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant"
        assert isinstance(assistant_message['content'], list)
        last_text_part = assistant_message['content'][-1]['text']
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        return int(pred_num == ref_num)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))


if __name__ == "__main__":
    print("Loading HindiGSM8K...")
    train = HindiGSM8K(split="train")
    test = HindiGSM8K(split="test")
    print(f"Train: {len(train)}, Test: {len(test)}")

    print("\nFirst 2 examples:")
    for i in range(2):
        ex = train[i]
        print("=" * 80)
        print(f"Q: {ex['messages'][0]['content'][:200]}")
        parts = ex['messages'][1]['content']
        rendered = ""
        for p in parts:
            if p['type'] == 'text':
                rendered += p['text']
            elif p['type'] == 'python':
                rendered += f"<<{p['text']}="
            elif p['type'] == 'python_output':
                rendered += f"{p['text']}>>"
        print(f"A: {rendered[:300]}")
