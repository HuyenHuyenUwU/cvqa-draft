"""Chuẩn hóa cường độ."""
import numpy as np
import torch


def min_max_unit(pixels: np.ndarray) -> np.ndarray:
    """Min-max về [0,1], KHÔNG kéo giãn tương phản (dùng cho Image Quality)."""
    value_range = float(pixels.max() - pixels.min())
    if value_range < 1e-6:
        return np.zeros_like(pixels, dtype=np.float32)
    return ((pixels - pixels.min()) / value_range).astype(np.float32)


def normalize_mean_std(image_chw: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    return (image_chw - mean) / std
