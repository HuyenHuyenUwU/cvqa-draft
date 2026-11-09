"""Answer Head = the LLM's own language-model output head (no fixed answer vocabulary).

It is kept separate from the decoder body so that
  Fusion (multimodal tokens) -> LLMDecoder (transformer body, hidden states) -> AnswerHead (token logits)
are three distinct, testable pieces:  P(A|Z) = prod_t P(a_t | a_<t, Z).
To save memory, logits are computed only at answer positions (the V=150k vocabulary projection over the
whole prompt is avoided). The loss itself lives in offline/losses.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AnswerHead(nn.Module):
    def __init__(self, lm_head: nn.Module):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor):
        """hidden [B, L, H]; labels [B, L] with -100 outside answer tokens.
        Returns (logits [M, V], targets [M]) for the M answer positions (next-token shifted)."""
        sel = labels[:, 1:] != -100
        h = hidden[:, :-1][sel]
        return self.lm_head(h), labels[:, 1:][sel]
