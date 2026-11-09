"""Type-Conditional Prompt.

Question type = RULE-BASED keyword classifier (a transparent heuristic, NOT a learned medical-intent model).
Independent of any encoder internals. Rules are checked in the listed order; first match wins.
  in : question text, entity-linking result
  out: {"question_type", "type_id", "prompt_text", "method": "rule_based_keywords", "matched": [...]}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..common.schemas import QuestionType as QT

RULES: List[Tuple[str, List[str]]] = [
    (QT.EXPLANATION, ["why", "explain", "reason", "because", "how do you know", "giải thích", "tại sao"]),
    (QT.ANATOMY_LOCATION, ["where", "which bone", "which part", "which region", "location", "located", "anatomical",
                           "site of", "ở đâu", "vị trí", "xương nào"]),
    (QT.FRACTURE_ID, ["fracture", "broken", "break", "crack", "gãy"]),
    (QT.RADIOGRAPHIC_SIGN, ["sign", "appearance", "density", "lucen", "sclerosis", "periosteal", "cortex", "cortical",
                            "margin", "pattern", "dấu hiệu"]),
    (QT.DISEASE_ID, ["disease", "tumor", "tumour", "lesion", "malignan", "benign", "diagnos", "osteo", "cancer",
                     "condition", "category", "finding", "bệnh", "u xương"]),
]

PROMPTS = {
    QT.ANATOMY_LOCATION: "Task: state the anatomical location of the finding in the image.",
    QT.FRACTURE_ID: "Task: decide whether a fracture is visible and answer concisely.",
    QT.DISEASE_ID: "Task: identify the finding or disease category suggested by the image.",
    QT.RADIOGRAPHIC_SIGN: "Task: describe the relevant radiographic sign visible in the image.",
    QT.EXPLANATION: "Task: explain the answer by referring to what is visible in the image.",
    QT.OTHER: "Task: answer the question about the image concisely.",
}


def classify_question_type(text: str) -> Tuple[str, List[str]]:
    t = text.lower()
    for qtype, kws in RULES:
        hit = [k for k in kws if re.search(r"\b" + re.escape(k), t)]
        if hit:
            return qtype, hit
    return QT.OTHER, []


class TypeConditionalPromptBuilder:
    def build(self, question: str, entity_result: Optional[Dict[str, Any]] = None,
              question_embedding=None) -> Dict[str, Any]:
        qtype, matched = classify_question_type(question)
        text = PROMPTS[qtype]
        ents = (entity_result or {}).get("entities", [])
        if ents:                                    # only real linked entities are ever added
            text += " Relevant concepts: " + ", ".join(str(e) for e in ents) + "."
        return {"question_type": qtype, "type_id": QT.to_id(qtype), "prompt_text": text,
                "method": "rule_based_keywords", "matched": matched}
