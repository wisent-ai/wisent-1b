"""Tokenizer for Rej-1B.

Provides a small, dependency-light character-level tokenizer by default.
An optional HuggingFace backend can be enabled if ``transformers`` is installed
and functional in the environment.
"""
from __future__ import annotations

from typing import List, Union


class RejTokenizer:
    """Character-level tokenizer with an optional HuggingFace backend.

    Args:
        vocab_size: maximum vocabulary size. For the native char tokenizer this
            limits the number of distinct characters kept.
        pretrained_name: if provided, try to load a HuggingFace tokenizer.
            Falls back to the native tokenizer on import/loading errors.
    """

    def __init__(self, vocab_size: int | None = None, pretrained_name: str | None = None):
        self.pretrained_name = pretrained_name
        self._hf_tokenizer = None
        self.vocab_size = vocab_size

        if pretrained_name is not None:
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(pretrained_name)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.model_max_length = int(1e9)
                self._hf_tokenizer = tok
                self.vocab_size = len(tok) if vocab_size is None else vocab_size
                return
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Warning: could not load HF tokenizer '{pretrained_name}': {exc}. "
                    "Falling back to native char tokenizer."
                )
                self._hf_tokenizer = None

        # Native char-level tokenizer.
        self._pad_id = 0
        self._eos_id = 1
        self._unk_id = 2
        # Build a deterministic character vocabulary of printable ASCII + common extras.
        chars = (
            ["<PAD>", "<EOS>", "<UNK>"]
            + [chr(i) for i in range(32, 127)]
            + ["\n", "\t"]
            + [chr(i) for i in range(161, 256)]  # extended latin
        )
        if vocab_size is not None:
            chars = chars[:vocab_size]
        self._id_to_char = chars
        self._char_to_id = {c: i for i, c in enumerate(chars)}
        if self.vocab_size is None:
            self.vocab_size = len(self._id_to_char)

    def encode(self, text: str, max_length: int | None = None) -> List[int]:
        if self._hf_tokenizer is not None:
            kwargs = {"add_special_tokens": False}
            if max_length is not None:
                kwargs["max_length"] = max_length
                kwargs["truncation"] = True
            ids = self._hf_tokenizer.encode(text, **kwargs)
            if self.vocab_size is not None:
                ids = [min(i, self.vocab_size - 1) for i in ids]
            return ids

        ids = [self._char_to_id.get(ch, self._unk_id) for ch in text]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> str:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if token_ids and isinstance(token_ids[0], list):
            token_ids = [item for sublist in token_ids for item in sublist]

        special = {self._pad_id, self._eos_id, self._unk_id} if skip_special_tokens else set()
        chars = []
        for idx in token_ids:
            idx = int(idx)
            if idx in special:
                continue
            if idx < len(self._id_to_char):
                chars.append(self._id_to_char[idx])
            else:
                chars.append("<?>")
        return "".join(chars)

    def batch_encode(self, texts: List[str], max_length: int | None = None) -> List[List[int]]:
        return [self.encode(t, max_length=max_length) for t in texts]

    @property
    def eos_token_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.eos_token_id
        return self._eos_id

    @property
    def pad_token_id(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.pad_token_id
        return self._pad_id
