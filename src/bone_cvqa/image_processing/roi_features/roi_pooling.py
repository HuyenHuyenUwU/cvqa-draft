"""Pooling đặc trưng patch theo mặt nạ ROI (trung bình có mặt nạ)."""
import torch


def masked_mean_pool(patch_features: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    """patch_features [B,N,D], patch_mask [B,N] -> [B,D]"""
    weights = patch_mask / patch_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.einsum("bn,bnd->bd", weights, patch_features)
