"""Batch collation. Missing lesion boxes are explicit (`has_box` mask), never zero-filled targets."""
from __future__ import annotations

from typing import Any, Dict, List

import torch


def cvqa_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    boxes, has_box = [], []
    for s in samples:
        la = s.get("lesion_annotation") or {}
        ok = la.get("status") == "available" and la.get("box_xyxy_norm") is not None
        has_box.append(ok)
        boxes.append(la["box_xyxy_norm"] if ok else [0.0, 0.0, 0.0, 0.0])
    answers = [s.get("answer") for s in samples]
    return {
        "sample_id": [s["sample_id"] for s in samples],
        "image": torch.stack([s["image"] for s in samples]),
        "question": [s["question"] for s in samples],
        "conversation_history": [s.get("conversation_history", []) for s in samples],
        "answer": answers,
        "has_answer": torch.tensor([a is not None for a in answers]),
        "gt_box": torch.tensor(boxes, dtype=torch.float32),     # [B,4] xyxy normalised
        "has_box": torch.tensor(has_box, dtype=torch.bool),     # [B]
        "metadata": [s.get("metadata", {}) for s in samples],
    }
