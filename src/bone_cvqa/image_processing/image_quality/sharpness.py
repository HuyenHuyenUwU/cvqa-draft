"""Độ nét: phương sai của Laplacian (heuristic; phụ thuộc giải phẫu và độ phân giải)."""
import cv2
import numpy as np


def laplacian_variance(gray_unit: np.ndarray) -> float:
    return float(cv2.Laplacian((gray_unit * 255).astype(np.uint8), cv2.CV_64F).var())
