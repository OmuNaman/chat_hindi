"""
Hindi MMLU dataset (MILU by AI4Bharat).
https://huggingface.co/datasets/ai4bharat/MILU

India-specific knowledge QA across 8 domains, 41 subjects.
Culturally grounded, not just translations of English MMLU.

Fallback: openai/MMMLU with HI_IN locale if MILU fails to load.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc


class HindiMMLU(Task):
    """Hindi multiple-choice knowledge QA."""

    letters = ('A', 'B', 'C', 'D')

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "HindiMMLU split must be train|test"
        self.split = split

        try:
            # Try MILU first (better quality, India-specific)
            if split == "train":
                self.ds = load_dataset("ai4bharat/MILU", "Hindi", split="validation").shuffle(seed=42)
            else:
                self.ds = load_dataset("ai4bharat/MILU", "Hindi", split="test").shuffle(seed=42)
            self._source = "MILU"
            # MILU format: question, options (list), answer (letter)
            self._format = "milu"
        except Exception:
            # Fallback to MMMLU
            ds = load_dataset("openai/MMMLU", "HI_IN", split="test").shuffle(seed=42)
            total = len(ds)
            split_idx = int(total * 0.8)
            if split == "train":
                self.ds = ds.select(range(split_idx))
            else:
                self.ds = ds.select(range(split_idx, total))
            self._source = "MMMLU"
            self._format = "mmmlu"

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]

        if self._format == "milu":
            question = row["question"]
            choices = row["options"]  # list of 4 strings
            answer_letter = row["answer"]  # "A", "B", "C", or "D"
            answer_idx = self.letters.index(answer_letter)
        else:
            # MMMLU format
            question = row["Question"]
            choices = [row["A"], row["B"], row["C"], row["D"]]
            answer_letter = row["Answer"]
            answer_idx = self.letters.index(answer_letter)

        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer_idx]

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        return {
            "messages": messages,
            "letters": self.letters,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, \
            f"Expected one of {self.letters}, got {assistant_response}"
        return assistant_response == conversation['messages'][-1]['content']


if __name__ == "__main__":
    print("Loading HindiMMLU...")
    task = HindiMMLU(split="train")
    print(f"Source: {task._source}, Size: {len(task)}")

    print("\nFirst 3 examples:")
    for i in range(3):
        ex = task[i]
        print("=" * 80)
        print(f"User: {ex['messages'][0]['content'][:300]}")
        print(f"Assistant: {ex['messages'][1]['content']}")
