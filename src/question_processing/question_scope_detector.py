"""Question Scope Detector (Online only) - transparent rule-based BASELINE, not a trained classifier.

Policy while the ontology is a placeholder (documented fallback):
  1. If the ontology-linking result contains entities  -> in_scope (reason: linked entities).
  2. Else if the question matches a configurable bone/radiology keyword list -> in_scope.
  3. Else -> ``unmatched_policy`` (default ``out_of_scope``) with a reason that states the limitation:
     the keyword list is incomplete, so this may be a false rejection.
Matching: whole words (plural 's'/'es' allowed); a keyword ending in '*' is a stem (prefix) match.
A follow-up question with no keyword is in_scope if an earlier turn of the conversation matched.
out : {"label": "in_scope"|"out_of_scope", "status": "available", "method": ..., "reason": str|None}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..common.schemas import Status

DEFAULT_KEYWORDS = [
    "bone", "fracture", "broken", "x-ray", "xray", "radiograph", "lesion", "tumor", "tumour", "joint", "femur",
    "tibia", "fibula", "humerus", "radius", "ulna", "hip", "shoulder", "knee", "ankle", "wrist", "elbow", "spine",
    "skull", "rib", "pelvis", "cortex", "cortical", "osteo*", "cartilage", "sclerosis", "periosteal", "benign",
    "malignant", "image", "scan", "finding", "hand", "leg", "arm", "foot", "this region", "xương", "gãy", "ảnh",
    "x-quang", "tổn thương", "khối u",
]


class RuleBasedQuestionScopeDetector:
    def __init__(self, keywords: Optional[List[str]] = None, unmatched_policy: str = "out_of_scope"):
        assert unmatched_policy in ("out_of_scope", "in_scope")
        self.keywords = [k.lower() for k in (keywords or DEFAULT_KEYWORDS)]
        self.unmatched_policy = unmatched_policy

    def _hits(self, text: str) -> List[str]:
        t = text.lower()
        out = []
        for k in self.keywords:
            pat = re.escape(k[:-1]) + r"\w*" if k.endswith("*") else re.escape(k) + r"(?:e?s)?"
            if re.search(r"(?<!\w)" + pat + r"(?!\w)", t):
                out.append(k)
        return out

    def detect(self, question: str, entity_result: Optional[Dict[str, Any]] = None,
               ontology_available: bool = False, history: Optional[list] = None) -> Dict[str, Any]:
        ents = (entity_result or {}).get("entities", [])
        if ents:
            return self._r("in_scope", "linked ontology entities found", "entity_link")
        hit = self._hits(question)
        if hit:
            return self._r("in_scope", f"matched domain keyword(s): {hit[:3]}", "keyword_fallback")
        from ..data.text_processing import normalize_history
        past = " ".join(t["content"] for t in normalize_history(history))
        if past and self._hits(past):
            return self._r("in_scope", "follow-up question; earlier turns matched domain keywords", "keyword_history")
        reason = ("no bone/radiology keyword matched. Limitation: ontology entity linking is not implemented and the "
                  "keyword list is incomplete, so this may be a false rejection.")
        return self._r(self.unmatched_policy, reason, "keyword_fallback")

    @staticmethod
    def _r(label, reason, method):
        return {"label": label, "status": Status.AVAILABLE, "method": method, "reason": reason}
