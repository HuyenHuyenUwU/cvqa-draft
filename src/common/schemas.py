"""Shared status vocabulary and result containers.

Every optional / postponed module returns a dict that contains a ``status`` field
taken from :class:`Status`. A placeholder NEVER returns fabricated values.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class Status:
    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_AVAILABLE = "not_available"  # implemented but no checkpoint/data (e.g. localizer untrained)
    SKIPPED = "skipped"
    FAILED = "failed"


class QuestionType:
    ANATOMY_LOCATION = "anatomy_location"
    FRACTURE_ID = "fracture_identification"
    DISEASE_ID = "disease_identification"
    RADIOGRAPHIC_SIGN = "radiographic_sign"
    EXPLANATION = "explanation"
    OTHER = "other"

    ALL: List[str] = [ANATOMY_LOCATION, FRACTURE_ID, DISEASE_ID, RADIOGRAPHIC_SIGN, EXPLANATION, OTHER]

    @classmethod
    def to_id(cls, name: str) -> int:
        return cls.ALL.index(name) if name in cls.ALL else cls.ALL.index(cls.OTHER)


# Annotation provenance (never mix silently)
ANSWER_SOURCE_EXPERT = "expert_annotation"
ANSWER_SOURCE_WEAK = "template_weak_supervision"

# ---- Sample contract (dataset -> model) ------------------------------------
# {
#   "sample_id": str,
#   "image": Tensor[3,H,W],
#   "question": str,
#   "conversation_history": List[{"question": str, "answer": str}],
#   "answer": Optional[str],
#   "lesion_annotation": {"status": "available"|"missing", "box_xyxy_norm": Optional[List[float]], ...},
#   "metadata": dict,
# }
SampleDict = Dict[str, Any]


def missing_lesion() -> Dict[str, Any]:
    return {"status": "missing", "box_xyxy_norm": None, "format": "xyxy_normalized", "source": None}


def placeholder_result(**fields: Any) -> Dict[str, Any]:
    out = dict(fields)
    out.setdefault("status", Status.NOT_IMPLEMENTED)
    return out


def opt(d: Optional[dict], key: str, default=None):
    return default if d is None else d.get(key, default)
