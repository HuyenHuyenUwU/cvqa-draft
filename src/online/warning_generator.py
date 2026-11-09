"""Warning Generator (Trustworthy Verification, rule-based).

in : image quality result, question scope result, uncertainty result, confidence result
out: list of {"code", "severity", "message"}
A low-confidence warning is produced ONLY when a validated confidence method produced a real number
(status == available and score is not None). Missing confidence yields an informational notice instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..common.schemas import Status


class WarningGenerator:
    def __init__(self, confidence_threshold: float = 0.5):
        self.conf_thr = confidence_threshold

    def generate(self, image_quality: Optional[Dict[str, Any]], question_scope: Optional[Dict[str, Any]],
                 uncertainty: Optional[Dict[str, Any]], confidence: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        w: List[Dict[str, str]] = []
        iq = image_quality or {}
        if iq.get("label") == "invalid":
            w.append({"code": "invalid_image", "severity": "high",
                      "message": f"The input image is invalid or unusable ({', '.join(iq.get('warnings', []))})."})
        elif iq.get("label") == "low":
            w.append({"code": "low_image_quality", "severity": "medium",
                      "message": "Image quality may be insufficient for a reliable answer: "
                                 + ", ".join(iq.get("warnings", [])) + " (heuristic check, not clinical-grade)."})
        sc = question_scope or {}
        if sc.get("label") == "out_of_scope":
            w.append({"code": "out_of_scope_question", "severity": "high",
                      "message": "The question appears to be outside the supported bone X-ray scope; no answer was generated. "
                                 + (sc.get("reason") or "")})
        if (uncertainty or {}).get("status") != Status.AVAILABLE:
            w.append({"code": "uncertainty_unavailable", "severity": "info",
                      "message": "Uncertainty estimation is not implemented yet; the answer carries no reliability estimate."})
        c = confidence or {}
        if c.get("status") == Status.AVAILABLE and c.get("score") is not None and c["score"] < self.conf_thr:
            w.append({"code": "low_confidence", "severity": "medium",
                      "message": f"Model confidence {c['score']:.2f} is below the threshold {self.conf_thr:.2f}."})
        return w
