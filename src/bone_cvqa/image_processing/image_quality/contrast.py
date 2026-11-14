"""Tương phản, độ sáng, độ bão hòa."""
import numpy as np


def percentile_contrast(gray_unit: np.ndarray) -> float:
    p5, p95 = np.percentile(gray_unit, [5, 95])
    return float(p95 - p5)


def mean_intensity(gray_unit: np.ndarray) -> float:
    return float(gray_unit.mean())


def saturated_fraction(gray_unit: np.ndarray) -> float:
    return float(((gray_unit <= 0.01) | (gray_unit >= 0.99)).mean())
