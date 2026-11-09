"""Tiny RANDOMLY-INITIALISED stand-ins for BiomedCLIP / LLM, built from the REAL open_clip and
transformers classes (TimmModel, HFTextEncoder, Qwen2ForCausalLM). They exist for unit tests, the CPU
smoke test and debugging; they download nothing and produce meaningless answers."""
from __future__ import annotations

import zlib
from typing import List

import torch
import torch.nn as nn


class TinyBiomedCLIP(nn.Module):
    """Same attribute layout as open_clip.CustomTextCLIP: .visual (TimmModel) and .text (HFTextEncoder)."""

    def __init__(self, embed_dim=32, image_size=64, vocab=512, text_hidden=48):
        super().__init__()
        from open_clip.hf_model import HFTextEncoder
        from open_clip.timm_model import TimmModel
        from transformers import BertConfig
        self.visual = TimmModel("vit_tiny_patch16_224", embed_dim, image_size=image_size, pool="", proj="linear",
                                pretrained=False, model_kwargs={"img_size": image_size})
        cfg = BertConfig(vocab_size=vocab, hidden_size=text_hidden, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, pad_token_id=0, max_position_embeddings=128)
        self.text = HFTextEncoder("", embed_dim, config=cfg, pooler_type="cls_pooler", proj_type="linear",
                                  pretrained=False)


def _hash_ids(text: str, vocab: int) -> List[int]:
    return [3 + zlib.crc32(w.encode()) % (vocab - 3) for w in text.lower().split()]


class DummyClipTokenizer:
    """open_clip-style: list[str] -> LongTensor [B, context_length], CLS id=1, pad id=0."""

    def __init__(self, vocab=512, context_length=96):
        self.vocab, self.context_length = vocab, context_length

    def __call__(self, texts):
        out = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = ([1] + _hash_ids(t, self.vocab))[: self.context_length]
            out[i, : len(ids)] = torch.tensor(ids)
        return out


class DummyLLMTokenizer:
    """HF-tokenizer-like subset used by LLMDecoder."""
    eos_token_id, pad_token_id, pad_token = 2, 0, "<pad>"

    def __init__(self, vocab=512):
        self.vocab, self._rev = vocab, {}

    def __call__(self, text, add_special_tokens=False):
        ids = _hash_ids(text, self.vocab)
        for w, i in zip(text.lower().split(), ids):
            self._rev[i] = w
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(self._rev.get(i, f"<{i}>") for i in ids if not (skip_special_tokens and i in (0, 1, 2)))


def build_tiny_llm(vocab=512, hidden=64):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=2, pad_token_id=0, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg)


def build_tiny_components(image_size=64, vocab=512, seed=0):
    torch.manual_seed(seed)
    clip = TinyBiomedCLIP(image_size=image_size, vocab=vocab)
    return clip, DummyClipTokenizer(vocab), build_tiny_llm(vocab), DummyLLMTokenizer(vocab)
