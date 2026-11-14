"""Warning Generator — CHỈ CÓ Ở ONLINE. Luật đơn giản.
Cảnh báo low_confidence CHỈ phát khi confidence có status=available và là số thật."""
from typing import Any, Dict, List, Optional

from ..commons.constants import Status

ONLINE_ONLY = True


class WarningGenerator:
    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def generate(self, image_quality: Optional[Dict[str, Any]], question_scope: Optional[Dict[str, Any]],
                 uncertainty: Optional[Dict[str, Any]], confidence: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        warnings: List[Dict[str, str]] = []
        quality = image_quality or {}
        if quality.get("label") == "invalid":
            warnings.append({"code": "invalid_image", "severity": "high",
                             "message": f"The input image is invalid or unusable ({', '.join(quality.get('warnings', []))})."})
        elif quality.get("label") == "low":
            warnings.append({"code": "low_image_quality", "severity": "medium",
                             "message": "Image quality may be insufficient: " + ", ".join(quality.get("warnings", []))
                                        + " (heuristic check, not clinical-grade)."})
        scope = question_scope or {}
        if scope.get("label") == "out_of_scope":
            warnings.append({"code": "out_of_scope_question", "severity": "high",
                             "message": "The question appears outside the supported bone X-ray scope; no answer was generated. "
                                        + (scope.get("reason") or "")})
        if (uncertainty or {}).get("status") != Status.AVAILABLE:
            warnings.append({"code": "uncertainty_unavailable", "severity": "info",
                             "message": "Uncertainty estimation is not implemented yet; the answer carries no reliability estimate."})
        conf = confidence or {}
        if conf.get("status") == Status.AVAILABLE and conf.get("score") is not None and conf["score"] < self.confidence_threshold:
            warnings.append({"code": "low_confidence", "severity": "medium",
                             "message": f"Model confidence {conf['score']:.2f} is below the threshold {self.confidence_threshold:.2f}."})
        return warnings
