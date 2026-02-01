"""
CustomJSON task for loading conversations from JSONL files.
Each line should be a JSON array of message objects with 'role' and 'content'.

Example line:
[{"role":"user","content":"तुम कौन हो?"},{"role":"assistant","content":"मैं नैनो हिंदी हूँ।"}]

Ported from nanochat.
"""

import os
import json
from tasks.common import Task


class CustomJSON(Task):
    """Load conversations from a JSONL file."""

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        if not os.path.exists(filepath):
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("-" * 80)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    messages = json.loads(line)
                    assert isinstance(messages, list), f"Expected list, got {type(messages)}"
                    assert len(messages) >= 2, f"Need at least 2 messages, got {len(messages)}"
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Message {i} missing 'role'"
                        assert "content" in message, f"Message {i} missing 'content'"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, \
                            f"Message {i} role={message['role']}, expected {expected_role}"
                        assert isinstance(message["content"], str), "Content must be a string"
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        return {"messages": messages}
