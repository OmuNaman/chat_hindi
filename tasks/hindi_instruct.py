"""
Hindi instruction-following dataset from AI4Bharat.
https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1

~385K quality-filtered Hindi instructions from:
Dolly, OASST1, HH-RLHF, FLAN v2, WikiHow, Anudesh, LMSys-Chat.
Translations filtered by chrF++ back-translation score >= 50.
"""

from datasets import load_dataset
from tasks.common import Task


class HindiInstruct(Task):
    """
    ai4bharat/indic-instruct-data-v0.1 dataset.
    Instruction/input/output format converted to user/assistant conversations.
    """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "HindiInstruct split must be train|test"
        self.split = split

        # Load the dataset - it has a single 'train' split, we split ourselves
        ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", split="train")
        ds = ds.shuffle(seed=42)

        # 95/5 train/test split
        total = len(ds)
        split_idx = int(total * 0.95)

        if split == "train":
            self.ds = ds.select(range(split_idx))
        else:
            self.ds = ds.select(range(split_idx, total))

        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        # Build user message from instruction + optional input
        instruction = row.get("instruction", "") or ""
        input_text = row.get("input", "") or ""
        output_text = row.get("output", "") or ""

        if input_text.strip():
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        messages = [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": output_text.strip()},
        ]

        return {"messages": messages}


if __name__ == "__main__":
    print("Loading HindiInstruct train split...")
    task = HindiInstruct(split="train")
    print(f"Train size: {len(task)}")

    print("\nFirst 3 examples:")
    for i in range(3):
        ex = task[i]
        print("=" * 80)
        print(f"User: {ex['messages'][0]['content'][:200]}...")
        print(f"Assistant: {ex['messages'][1]['content'][:200]}...")
