"""Project đặc trưng backbone về d_model dùng chung (Linear + LayerNorm)."""
import torch
from torch import nn


class FeatureProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(features))
