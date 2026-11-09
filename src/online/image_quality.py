"""Image Quality Assessment (lightweight HEURISTIC baseline).

in : PreprocessedImage (uses ``gray_minmax``: min-max scaled, no contrast stretching) + original size
out: {"status": "available", "label": "good"|"low"|"invalid", "score": float, "warnings": [...], "metrics": {...}}
``score`` = fraction of heuristic checks passed (0..1). It is NOT a calibrated or clinical-grade quality
measure: thresholds are configurable and untuned, and blur (variance of Laplacian) depends on anatomy/resolution.
"label=invalid" means empty / constant image; "low" means at least one check failed.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..common.schemas import Status

DEFAULTS = {"min_std": 0.02, "min_contrast_p95_p5": 0.15, "max_mean": 0.92, "min_mean": 0.05,
            "max_saturated_fraction": 0.5, "min_blur_laplacian_var": 20.0, "min_side_px": 128}


class ImageQualityAssessor:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.t = {**DEFAULTS, **(thresholds or {})}

    def assess(self, gray01: Optional[np.ndarray], original_size_hw=None) -> Dict[str, Any]:
        if gray01 is None or gray01.size == 0 or not np.isfinite(gray01).all():
            return {"status": Status.AVAILABLE, "label": "invalid", "score": 0.0,
                    "warnings": ["invalid_or_unreadable_image"], "metrics": {}}
        t, g = self.t, gray01.astype(np.float32)
        std = float(g.std())
        if std < t["min_std"]:
            return {"status": Status.AVAILABLE, "label": "invalid", "score": 0.0,
                    "warnings": ["empty_or_constant_image"], "metrics": {"std": std}}
        p5, p95 = np.percentile(g, [5, 95])
        lap = float(cv2.Laplacian((g * 255).astype(np.uint8), cv2.CV_64F).var())
        sat = float(((g <= 0.01) | (g >= 0.99)).mean())
        mean = float(g.mean())
        checks = {
            "low_contrast": (p95 - p5) >= t["min_contrast_p95_p5"],
            "under_exposed": mean >= t["min_mean"],
            "over_exposed": mean <= t["max_mean"],
            "extreme_intensity_distribution": sat <= t["max_saturated_fraction"],
            "possible_blur": lap >= t["min_blur_laplacian_var"],
            "low_resolution": original_size_hw is None or min(original_size_hw) >= t["min_side_px"],
        }
        warns = [k for k, ok in checks.items() if not ok]
        return {"status": Status.AVAILABLE, "label": "low" if warns else "good",
                "score": sum(checks.values()) / len(checks), "warnings": warns,
                "metrics": {"std": std, "contrast_p95_p5": float(p95 - p5), "mean": mean, "saturated_fraction": sat,
                            "laplacian_var": lap, "heuristic": True}}
