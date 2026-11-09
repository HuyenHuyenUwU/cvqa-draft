"""Final Output: one consistent response schema. Unavailable outputs are explicit, never invented."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..common.schemas import Status

REQUIRED_KEYS = ["sample_id", "answer", "localization", "evidence_graph", "visual_knowledge_explanation", "uncertainty",
                 "confidence", "image_quality", "question_scope", "warnings", "pipeline_status", "disclaimer"]
DISCLAIMER = "Decision-support output only; it does not replace a clinician's judgement."


def localization_block(loc: Optional[Dict[str, Any]], trained: bool, include_heatmap: bool = False,
                       grid_size=None, reason: Optional[str] = None) -> Dict[str, Any]:
    if loc is None or not trained:
        return {"type": None, "value": None, "status": Status.NOT_AVAILABLE,
                "reason": reason or ("localizer has no trained checkpoint" if loc is not None else "localizer unavailable")}
    out = {"type": "bounding_box", "value": [round(float(v), 4) for v in loc["box_xyxy_norm"][0].tolist()],
           "format": "xyxy_normalized_to_original_image", "status": Status.AVAILABLE, "source": "predicted"}
    if include_heatmap and grid_size:
        h = loc["heatmap_logits"][0].softmax(-1).reshape(*grid_size)
        out["heatmap"] = {"grid_size": list(grid_size), "values": [[round(float(x), 5) for x in r] for r in h]}
    return out


def format_output(sample_id: str, answer: Optional[str], localization: Dict[str, Any], evidence_graph: Dict[str, Any],
                  explanation: Dict[str, Any], uncertainty: Dict[str, Any], confidence: Dict[str, Any],
                  image_quality: Dict[str, Any], question_scope: Dict[str, Any], warnings: List[Dict[str, str]],
                  pipeline_status: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"sample_id": sample_id, "answer": answer, "localization": localization, "evidence_graph": evidence_graph,
           "visual_knowledge_explanation": explanation, "uncertainty": uncertainty, "confidence": confidence,
           "image_quality": image_quality, "question_scope": question_scope, "warnings": warnings,
           "pipeline_status": pipeline_status, "disclaimer": DISCLAIMER}
    if extra:
        out["debug"] = extra
    return out


def validate_output_schema(out: Dict[str, Any]) -> List[str]:
    """Returns the list of problems (empty = valid)."""
    probs = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in out]
    for k in ("localization", "evidence_graph", "visual_knowledge_explanation", "uncertainty", "confidence",
              "image_quality", "question_scope"):
        if k in out and "status" not in out[k]:
            probs.append(f"{k} has no status")
    if out.get("confidence", {}).get("status") != Status.AVAILABLE and out.get("confidence", {}).get("score") is not None:
        probs.append("confidence score present although status is not available (fabricated?)")
    if out.get("uncertainty", {}).get("status") != Status.AVAILABLE and out.get("uncertainty", {}).get("score") is not None:
        probs.append("uncertainty score present although status is not available (fabricated?)")
    return probs
