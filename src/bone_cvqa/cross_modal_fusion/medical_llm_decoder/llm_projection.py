"""Chiếu token đã fuse (d_model) sang không gian embedding của LLM (hidden_size đọc từ LLM, không hard-code)."""
from torch import nn


class FusionToLLMProjection(nn.Module):
    def __init__(self, d_model: int, llm_hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, llm_hidden_size)

    def forward(self, fused_tokens):
        return self.linear(fused_tokens)
