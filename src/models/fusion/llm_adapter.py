"""LLM adapter: fused multimodal tokens -> ``n_soft_tokens`` soft tokens in the LLM embedding space.

  in : tokens [B, K, d_model], mask [B, K]
  out: soft_tokens [B, n, H_llm]   (H_llm read from the LLM, never hard-coded)

Learned queries cross-attend to the fused tokens (Q-Former-lite), then a Linear maps to the LLM
hidden size. The soft tokens are PREPENDED to the prompt's token embeddings and fed to the LLM through
``inputs_embeds`` (see LLMDecoder) - standard text-generation APIs never see raw multimodal vectors.

Gradient flow: with a frozen LLM, gradients still flow *through* the LLM activations back into the
adapter (and fusion / unfrozen encoders); only adapter-side parameters receive updates.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LLMAdapter(nn.Module):
    def __init__(self, d_model: int, llm_hidden: int, n_soft_tokens: int = 8, n_heads: int = 4,
                 init_scale: float = 1.0):
        super().__init__()
        self.n_soft_tokens, self.llm_hidden = n_soft_tokens, llm_hidden
        self.queries = nn.Parameter(torch.randn(n_soft_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.to_llm = nn.Linear(d_model, llm_hidden)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=~mask, need_weights=False)
        return self.to_llm(self.norm(out + q)) * self.scale
