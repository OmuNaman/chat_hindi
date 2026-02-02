"""
Tokenizer wrapper for nano_hindi SFT.

Wraps Sarvam-1 tokenizer with conversation rendering and loss masking.

Chat template:
    <s>### उपयोगकर्ता:
    {user_message}

    ### सहायक:
    {assistant_response}</s>

Multi-turn conversations repeat user/assistant blocks.
Loss mask: 0 for user/system tokens, 1 for assistant tokens + EOS.
"""

import torch
from transformers import AutoTokenizer

TOKENIZER_NAME = "sarvamai/sarvam-1"

# Chat template markers
USER_MARKER = "### उपयोगकर्ता:\n"
ASSISTANT_MARKER = "### सहायक:\n"

_cached_tokenizer = None


def get_tokenizer():
    """Get cached Sarvam-1 tokenizer."""
    global _cached_tokenizer
    if _cached_tokenizer is None:
        _cached_tokenizer = SarvamTokenizer()
    return _cached_tokenizer


class SarvamTokenizer:
    """Sarvam-1 tokenizer with conversation rendering for SFT."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self._bos_id = self.tokenizer.bos_token_id  # 1
        self._eos_id = self.tokenizer.eos_token_id  # 2
        self._vocab_size = self.tokenizer.vocab_size  # 68096

    def get_bos_token_id(self):
        return self._bos_id

    def get_eos_token_id(self):
        return self._eos_id

    def get_vocab_size(self):
        return self._vocab_size

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def render_conversation(self, conversation):
        """
        Render a conversation into token IDs and a loss mask.

        Args:
            conversation: dict with "messages" key containing list of
                         {"role": "user"|"assistant"|"system", "content": str|list}

        Returns:
            (token_ids, mask): both lists of ints.
                token_ids: the full tokenized conversation including BOS/EOS.
                mask: 1 for tokens the model should predict (assistant + EOS), 0 otherwise.
        """
        messages = conversation["messages"]
        token_ids = [self._bos_id]
        mask = [0]  # BOS is not predicted

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # System messages are context, not predicted
                text = content + "\n\n"
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                token_ids.extend(ids)
                mask.extend([0] * len(ids))

            elif role == "user":
                text = USER_MARKER + content + "\n\n"
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                token_ids.extend(ids)
                mask.extend([0] * len(ids))

            elif role == "assistant":
                # Marker is not predicted, but the response IS predicted
                marker_ids = self.tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
                token_ids.extend(marker_ids)
                mask.extend([0] * len(marker_ids))

                # Handle content: can be a string or a list of parts (for tool calls)
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    # Render parts: text, python (tool call), python_output
                    response_text = self._render_parts(content)
                else:
                    response_text = str(content)

                response_text = response_text + "\n\n"
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                token_ids.extend(response_ids)
                mask.extend([1] * len(response_ids))  # Assistant tokens ARE predicted

        # Add EOS at the end (predicted)
        token_ids.append(self._eos_id)
        mask.append(1)

        return token_ids, mask

    def render_for_completion(self, conversation):
        """
        Render a conversation for RL completion generation.

        Strips the last assistant message (the reference answer) and returns
        token IDs up to and including the assistant marker, so the model can
        generate the completion.

        Args:
            conversation: dict with "messages" key. The last message must be
                         role="assistant" (its content is not included).

        Returns:
            prompt_ids: list of token IDs (BOS + messages + final assistant marker)
        """
        import copy
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be assistant"
        messages.pop()  # Remove last assistant message (reference answer)

        # Render remaining messages
        token_ids = [self._bos_id]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                ids = self.tokenizer.encode(content + "\n\n", add_special_tokens=False)
                token_ids.extend(ids)
            elif role == "user":
                ids = self.tokenizer.encode(USER_MARKER + content + "\n\n", add_special_tokens=False)
                token_ids.extend(ids)
            elif role == "assistant":
                # Non-final assistant messages: include marker + content
                marker_ids = self.tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
                token_ids.extend(marker_ids)
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = self._render_parts(content)
                else:
                    text = str(content)
                token_ids.extend(self.tokenizer.encode(text + "\n\n", add_special_tokens=False))

        # Prime the assistant for completion
        marker_ids = self.tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
        token_ids.extend(marker_ids)
        return token_ids

    def _render_parts(self, parts):
        """Render a list of content parts (text, python tool calls) into a string."""
        rendered = []
        for part in parts:
            ptype = part.get("type", "text")
            text = part.get("text", "")
            if ptype == "text":
                rendered.append(text)
            elif ptype == "python":
                # Tool call: <<expression=
                rendered.append(f"<<{text}=")
            elif ptype == "python_output":
                # Tool output: result>>
                rendered.append(f"{text}>>")
        return "".join(rendered)


def compute_token_bytes(device="cpu"):
    """
    Compute byte length for each token in the vocabulary.

    Returns:
        1D tensor of shape (vocab_size,) with byte counts.
        Special tokens have value 0 (excluded from BPB calculation).
    """
    tokenizer = get_tokenizer()
    tok = tokenizer.tokenizer
    vocab_size = tok.vocab_size
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32)

    special_token_ids = set()
    if hasattr(tok, "all_special_ids"):
        special_token_ids = set(tok.all_special_ids)

    for token_id in range(vocab_size):
        if token_id in special_token_ids:
            token_bytes[token_id] = 0
            continue
        try:
            text = tok.decode([token_id])
            token_bytes[token_id] = len(text.encode("utf-8"))
        except Exception:
            token_bytes[token_id] = 0

    return token_bytes.to(device)
