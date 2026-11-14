"""Tokenizer giả (băm từ -> id) chỉ để TEST OFFLINE khi không có vocab thật. Giao diện giống HuggingFace.

Id đặc biệt: pad=0, cls/bos=1, sep/eos=2.
"""
import zlib
from typing import Any, Dict, List, Union

import torch


class TinyHashTokenizer:
    pad_token_id, cls_token_id, eos_token_id = 0, 1, 2
    pad_token = "<pad>"

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self._reverse: Dict[int, str] = {}

    def _word_ids(self, text: str) -> List[int]:
        ids = []
        for word in text.lower().split():
            token_id = 3 + zlib.crc32(word.encode()) % (self.vocab_size - 3)
            self._reverse[token_id] = word
            ids.append(token_id)
        return ids

    def _encode(self, text: str, add_special_tokens: bool) -> List[int]:
        ids = self._word_ids(text)
        return [self.cls_token_id] + ids + [self.eos_token_id] if add_special_tokens else ids

    def __call__(self, text: Union[str, List[str]], add_special_tokens: bool = True, padding: bool = False,
                 truncation: bool = False, max_length: int = None, return_tensors: str = None) -> Dict[str, Any]:
        if isinstance(text, str):
            return {"input_ids": self._encode(text, add_special_tokens)}
        sequences = [self._encode(t, add_special_tokens) for t in text]
        if truncation and max_length:
            sequences = [s[:max_length] for s in sequences]
        longest = max(len(s) for s in sequences) if padding else None
        input_ids, masks = [], []
        for s in sequences:
            pad = (longest - len(s)) if longest else 0
            input_ids.append(s + [self.pad_token_id] * pad)
            masks.append([1] * len(s) + [0] * pad)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(masks)}
        return {"input_ids": input_ids, "attention_mask": masks}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(self._reverse.get(i, f"<{i}>") for i in ids if not (skip_special_tokens and i in (0, 1, 2)))
