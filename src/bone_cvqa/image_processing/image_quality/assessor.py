"""Image Quality Assessment — CHỈ CÓ Ở ONLINE (không có trong sơ đồ Offline; Offline không import module này).

Heuristic nhẹ, ngưỡng cấu hình được, KHÔNG phải đánh giá chất lượng lâm sàng.
Output: {"status", "label": good|low|invalid, "score": tỉ lệ kiểm tra đạt (0..1), "warnings": [...], "metrics": {...}}
"""
from typing import Any, Dict, Optional

import numpy as np

from ...commons.constants import Status
from .contrast import mean_intensity, percentile_contrast, saturated_fraction
from .noise import estimate_noise_sigma
from .sharpness import laplacian_variance

ONLINE_ONLY = True
DEFAULT_THRESHOLDS = {"min_std": 0.02, "min_contrast": 0.15, "min_mean": 0.05, "max_mean": 0.92,
                      "max_saturated_fraction": 0.5, "min_sharpness": 20.0, "max_noise_sigma": 0.1,
                      "min_side_px": 128}


class ImageQualityAssessor:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def assess(self, gray_unit: Optional[np.ndarray], original_size_hw=None) -> Dict[str, Any]:
        if gray_unit is None or gray_unit.size == 0 or not np.isfinite(gray_unit).all():
            return self._result("invalid", 0.0, ["invalid_or_unreadable_image"], {})
        t = self.thresholds
        std = float(gray_unit.std())
        if std < t["min_std"]:
            return self._result("invalid", 0.0, ["empty_or_constant_image"], {"std": std})
        metrics = {"std": std, "contrast": percentile_contrast(gray_unit), "mean": mean_intensity(gray_unit),
                   "saturated_fraction": saturated_fraction(gray_unit), "sharpness": laplacian_variance(gray_unit),
                   "noise_sigma": estimate_noise_sigma(gray_unit)}
        checks = {
            "low_contrast": metrics["contrast"] >= t["min_contrast"],
            "under_exposed": metrics["mean"] >= t["min_mean"],
            "over_exposed": metrics["mean"] <= t["max_mean"],
            "extreme_intensity_distribution": metrics["saturated_fraction"] <= t["max_saturated_fraction"],
            "possible_blur": metrics["sharpness"] >= t["min_sharpness"],
            "high_noise": metrics["noise_sigma"] <= t["max_noise_sigma"],
            "low_resolution": original_size_hw is None or min(original_size_hw) >= t["min_side_px"],
        }
        warnings = [name for name, passed in checks.items() if not passed]
        return self._result("low" if warnings else "good", sum(checks.values()) / len(checks), warnings, metrics)

    @staticmethod
    def _result(label: str, score: float, warnings, metrics) -> Dict[str, Any]:
        return {"status": Status.AVAILABLE, "label": label, "score": score, "warnings": warnings,
                "metrics": {**metrics, "heuristic": True}}
