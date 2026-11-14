"""Khối 6 (Online) — Trustworthy Verification: chất lượng ảnh + scope + uncertainty/confidence -> Warning Generator -> Final Output."""
from typing import Any, Dict, List, Optional

from .final_output import build_final_output
from .warning_generator import WarningGenerator

ONLINE_ONLY = True


class TrustworthyVerification:
    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.warning_generator = WarningGenerator(confidence_threshold)

    def finalize(self, sample_id: str, answer: Optional[str], placeholders: Dict[str, Dict[str, Any]],
                 uncertainty: Dict[str, Any], confidence: Dict[str, Any], image_quality: Dict[str, Any],
                 question_scope: Optional[Dict[str, Any]], pipeline_status: str,
                 debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        warnings = self.warning_generator.generate(image_quality, question_scope, uncertainty, confidence)
        scope = question_scope or {"label": None, "status": "skipped", "reason": "not evaluated"}
        return build_final_output(sample_id, answer, placeholders, uncertainty, confidence, image_quality, scope,
                                  warnings, pipeline_status, debug)
