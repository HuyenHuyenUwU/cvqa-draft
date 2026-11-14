"""Final Output — CHỈ CÓ Ở ONLINE. Một schema duy nhất; đầu ra không có là trạng thái tường minh, không bao giờ bịa."""
from typing import Any, Dict, List, Optional

from ..commons.constants import Status

ONLINE_ONLY = True
DISCLAIMER = "Decision-support output only; it does not replace a clinician's judgement."
REQUIRED_KEYS = ["sample_id", "answer", "localization", "evidence_graph", "visual_knowledge_explanation", "uncertainty",
                 "confidence", "image_quality", "question_scope", "warnings", "pipeline_status", "disclaimer"]


def build_final_output(sample_id: str, answer: Optional[str], placeholders: Dict[str, Dict[str, Any]],
                       uncertainty: Dict[str, Any], confidence: Dict[str, Any], image_quality: Dict[str, Any],
                       question_scope: Dict[str, Any], warnings: List[Dict[str, str]], pipeline_status: str,
                       debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    output = {"sample_id": sample_id, "answer": answer, "localization": placeholders["localization"],
              "evidence_graph": placeholders["evidence_graph"],
              "visual_knowledge_explanation": placeholders["visual_knowledge_explanation"],
              "uncertainty": uncertainty, "confidence": confidence, "image_quality": image_quality,
              "question_scope": question_scope, "warnings": warnings, "pipeline_status": pipeline_status,
              "disclaimer": DISCLAIMER}
    if debug:
        output["debug"] = debug
    return output


def validate_final_output(output: Dict[str, Any]) -> List[str]:
    """Trả danh sách lỗi (rỗng = hợp lệ). Bắt cả trường hợp bịa score khi status != available."""
    problems = [f"missing key: {key}" for key in REQUIRED_KEYS if key not in output]
    for key in ("localization", "evidence_graph", "visual_knowledge_explanation", "uncertainty", "confidence",
                "image_quality", "question_scope"):
        if key in output and "status" not in output[key]:
            problems.append(f"{key} has no status")
    for key in ("uncertainty", "confidence"):
        block = output.get(key, {})
        if block.get("status") != Status.AVAILABLE and block.get("score") is not None:
            problems.append(f"{key} score present although status is not available (fabricated?)")
    return problems
