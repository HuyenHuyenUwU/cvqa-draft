"""Lesion annotation helpers. Original annotation formats are preserved in metadata;
only a normalised enclosing box is exposed to the model."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .lesion import missing_lesion


def xywh_to_xyxy(b: Sequence[float]) -> List[float]:
    x, y, w, h = b
    return [x, y, x + w, y + h]


def points_to_xyxy(points: Sequence[Sequence[float]]) -> List[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_box(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def normalize_box(box: Sequence[float], size_wh: Tuple[int, int]) -> Optional[List[float]]:
    w, h = size_wh
    if w <= 0 or h <= 0:
        return None
    x1, y1, x2, y2 = box
    out = [max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)]
    return out if out[2] > out[0] and out[3] > out[1] else None


def make_lesion_annotation(boxes_px: Sequence[Sequence[float]], size_wh: Optional[Tuple[int, int]],
                           source: str) -> Dict[str, Any]:
    """Enclosing (union) normalised box over all annotated boxes; 'missing' if none."""
    if not boxes_px or size_wh is None:
        return missing_lesion()
    nb = normalize_box(union_box(boxes_px), size_wh)
    if nb is None:
        return missing_lesion()
    return {"status": "available", "box_xyxy_norm": nb, "format": "xyxy_normalized",
            "source": source, "num_boxes": len(boxes_px), "mask_available": False}
