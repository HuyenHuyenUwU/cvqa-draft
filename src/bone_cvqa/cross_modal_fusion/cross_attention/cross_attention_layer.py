"""Một lớp cross-attention: query (token câu hỏi) chú ý tới memory (đặc trưng ảnh/tri thức) + FFN, pre-LayerNorm."""
from typing import Optional

import torch
from torch import nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """queries [B,T,D]; memory [B,M,D]; memory_mask [B,M] bool (True = hợp lệ)."""
        key_padding_mask = None if memory_mask is None else ~memory_mask
        attended, _ = self.attention(self.query_norm(queries), memory, memory, key_padding_mask=key_padding_mask,
                                     need_weights=False)
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))
