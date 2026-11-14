"""Biến đổi hình học/kênh: resize vuông, gray -> 3 kênh."""
import numpy as np
import torch
from PIL import Image


def resize_square(gray_unit: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray((gray_unit * 255.0).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(image).astype(np.float32) / 255.0


def gray_to_three_channels(gray_unit: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(gray_unit).unsqueeze(0).repeat(3, 1, 1)
