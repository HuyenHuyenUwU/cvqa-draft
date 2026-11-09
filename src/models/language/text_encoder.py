"""Conversation-aware language encoder.

Backend "biomedclip" (default, as decided): BiomedCLIP's own text tower (open_clip HFTextEncoder,
a BERT initialised from PubMedBERT) is re-used -> text and image live in the same CLIP space.
  question_features  [B, T, Dt] : last hidden states, CLS position removed (Dt = BERT hidden size)
  question_embedding [B, De]    : CLS pooling + BiomedCLIP projection (CLIP text space)
Backend "hf" (fallback, standalone PubMedBERT checkpoint): AutoModel + mean pooling over valid tokens.

Input is the string produced by ``build_conversation_text`` ([HISTORY] ... [CURRENT QUESTION]).
Padding masks are derived from token ids != pad id and returned with the features.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn


@dataclass
class TextOutput:
    question_features: torch.Tensor     # [B, T, Dt]
    question_mask: torch.Tensor         # [B, T] bool (True = real token)
    question_embedding: torch.Tensor    # [B, De]


class ConversationTextEncoder(nn.Module):
    def __init__(self, backend: str, text_module: nn.Module, tokenizer: Any, max_length: int = 256,
                 pad_token_id: int = 0, freeze: bool = True):
        super().__init__()
        assert backend in ("biomedclip", "hf")
        self.backend, self.text, self.tokenizer = backend, text_module, tokenizer
        self.max_length, self.pad_id = int(max_length), int(pad_token_id)
        self._fully_frozen = freeze
        for p in self.text.parameters():
            p.requires_grad = not freeze
        if backend == "biomedclip":
            self.text.output_tokens = True
            self.feature_dim = self.text.config.hidden_size
            self.embedding_dim = self.text.output_dim
        else:
            self.feature_dim = self.embedding_dim = self.text.config.hidden_size
        self.train(self.training)          # a frozen tower must start in eval mode (no dropout)

    def train(self, mode: bool = True):
        super().train(mode and not self._fully_frozen)
        return self

    def _tokenize(self, texts: List[str], device):
        if self.backend == "biomedclip":
            ids = self.tokenizer(texts)                          # open_clip tokenizer -> [B, context_length]
            ids = ids[:, : self.max_length]
            mask = ids != self.pad_id
            keep = int(mask.sum(1).max().item())
            return ids[:, :keep].to(device), mask[:, :keep].to(device)
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].bool().to(device)

    def forward(self, texts: List[str]) -> TextOutput:
        device = next(self.text.parameters()).device
        ids, mask = self._tokenize(texts, device)
        ctx = torch.no_grad() if self._fully_frozen else nullcontext()
        with ctx:
            if self.backend == "biomedclip":
                emb, tokens = self.text(ids)
                pooler = getattr(self.text, "pooler", None)
                if type(pooler).__name__ == "ClsPooler":           # tokens exclude the CLS position
                    keep = torch.arange(ids.shape[1], device=device) != pooler.cls_token_position
                    mask = mask[:, keep]
                return TextOutput(tokens, mask, emb)
            out = self.text(input_ids=ids, attention_mask=mask.long()).last_hidden_state
            m = mask.unsqueeze(-1).to(out.dtype)
            pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)
            return TextOutput(out, mask, pooled)
