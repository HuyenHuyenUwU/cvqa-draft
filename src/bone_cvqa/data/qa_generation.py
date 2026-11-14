"""WEAK-SUPERVISION question/answer generation from image-level labels.

IMPORTANT: neither FracAtlas nor BTXRD is assumed to ship conversational QA. The pairs
produced here are *templates derived from existing labels/boxes*. They are tagged
``answer_source = "template_weak_supervision"`` and must never be reported as expert
ground truth. If an expert QA file exists use ``qa_source: file`` instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..commons.constants import ANSWER_SOURCE_EXPERT, ANSWER_SOURCE_WEAK


def coarse_region(box_norm: List[float]) -> str:
    cx, cy = (box_norm[0] + box_norm[2]) / 2, (box_norm[1] + box_norm[3]) / 2
    v = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    h = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    return f"{v}-{h} part of the image" if v != "middle" or h != "center" else "central part of the image"


def weak_qa_for_record(rec: Dict[str, Any], dataset: str, box_norm: Optional[List[float]]) -> List[Dict[str, Any]]:
    """Returns an ordered list of turns [{'question','answer','answer_source'}]."""
    label_name = rec.get("label_name")
    if label_name is None:
        return []
    turns: List[Dict[str, Any]] = []
    if dataset == "fracatlas":
        pos = int(rec["label"]) == 1
        turns.append({"question": "Is there a fracture in this X-ray image?",
                      "answer": "Yes, a fracture is present." if pos else "No, no fracture is annotated in this image."})
        if pos and box_norm is not None:
            turns.append({"question": "Where is the fracture located?",
                          "answer": f"The annotated fracture lies in the {coarse_region(box_norm)}."})
    else:  # btxrd and generic label datasets
        turns.append({"question": "What is the finding category of this bone X-ray?",
                      "answer": f"The annotated category is {label_name}."})
        if box_norm is not None:
            turns.append({"question": "Where is the lesion located?",
                          "answer": f"The annotated lesion lies in the {coarse_region(box_norm)}."})
    for t in turns:
        t["answer_source"] = ANSWER_SOURCE_WEAK
    return turns


def load_expert_qa(path) -> Dict[str, List[Dict[str, Any]]]:
    """JSONL with fields image_id, question, answer, (turn). Returns {image_id: ordered turns}."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault(str(r["image_id"]), []).append(
                {"question": r["question"], "answer": r.get("answer"), "answer_source": ANSWER_SOURCE_EXPERT,
                 "turn": r.get("turn", 0)})
    for v in out.values():
        v.sort(key=lambda t: t["turn"])
    return out
