"""Ước lượng nhiễu (Immerkaer 1996): sigma từ đáp ứng của kernel Laplacian 3x3 hiệu chỉnh."""
import cv2
import numpy as np

_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_noise_sigma(gray_unit: np.ndarray) -> float:
    height, width = gray_unit.shape
    if height < 3 or width < 3:
        return 0.0
    response = cv2.filter2D(gray_unit.astype(np.float64), -1, _KERNEL)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(response).sum() / (6.0 * (width - 2) * (height - 2)))
