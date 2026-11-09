"""Lesion Localization (LobA-style grounding branch, minimal baseline) + ROI Features.

LesionLocalizer
  in : patch_features [B, N, Dv]
  out: {"heatmap_logits": [B, N], "box_xyxy_norm": [B, 4]}  (box = enclosing lesion box, normalised)
  A learned patch-saliency head produces the heatmap; a box regressor reads the saliency-weighted
  patch summary plus the expected (x, y) of the saliency map. Supervision is the annotated box only
  (L1 + GIoU, plus an optional box-induced patch mask for the heatmap; NOT a segmentation mask).
  It is only meaningful after training on samples with ``has_box``; the model records this in
  ``localizer_trained`` and the online pipeline reports ``not_available`` otherwise.

RoIFeatureExtractor
  in : patch_features [B, N, Dv], grid_size (h, w), box_xyxy_norm [B, 4], box_valid [B] bool
  out: roi_features [B, Dv] (mean pooling of patches whose centre falls in the box; nearest patch if
       none does), roi_valid [B]. Rows without a box are zero + roi_valid=False (explicit missing).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def patch_centers(grid: Tuple[int, int], device) -> torch.Tensor:
    h, w = grid
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)           # [N, 2] (x, y)


def box_to_patch_mask(box: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    """box [B,4] xyxy-norm -> [B, N] float mask of patches whose centre lies inside (nearest patch fallback)."""
    c = patch_centers(grid, box.device)                                       # [N,2]
    inside = ((c[None, :, 0] >= box[:, None, 0]) & (c[None, :, 0] <= box[:, None, 2]) &
              (c[None, :, 1] >= box[:, None, 1]) & (c[None, :, 1] <= box[:, None, 3]))
    ctr = torch.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], -1)
    nearest = ((c[None] - ctr[:, None]) ** 2).sum(-1).argmin(-1)
    none = ~inside.any(1)
    if none.any():
        inside[none, nearest[none]] = True
    return inside.float()


class LesionLocalizer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.saliency = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.box_head = nn.Sequential(nn.Linear(in_dim + 2, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, patch_features: torch.Tensor, grid_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        logits = self.saliency(patch_features).squeeze(-1)                    # [B, N]
        w = logits.softmax(-1)
        pooled = torch.einsum("bn,bnd->bd", w, patch_features)
        exp_xy = torch.einsum("bn,nc->bc", w, patch_centers(grid_size, patch_features.device))
        cxcywh = self.box_head(torch.cat([pooled, exp_xy], -1)).sigmoid()
        cx, cy, bw, bh = cxcywh.unbind(-1)
        box = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1).clamp(0, 1)
        return {"heatmap_logits": logits, "box_xyxy_norm": box}


class RoIFeatureExtractor(nn.Module):
    def forward(self, patch_features: torch.Tensor, grid_size: Tuple[int, int],
                box_xyxy_norm: torch.Tensor, box_valid: torch.Tensor):
        m = box_to_patch_mask(box_xyxy_norm.detach(), grid_size)              # [B, N]
        pooled = torch.einsum("bn,bnd->bd", m, patch_features) / m.sum(1, keepdim=True).clamp_min(1.0)
        pooled = pooled * box_valid.unsqueeze(-1).to(pooled.dtype)
        return pooled, box_valid.clone()
