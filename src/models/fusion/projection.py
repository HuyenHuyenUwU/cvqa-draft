import torch.nn as nn


class ModalityProjection(nn.Module):
    """Explicit projection of one modality into the shared fusion width (Linear + LayerNorm)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))
