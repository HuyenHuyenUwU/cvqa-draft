"""Metrics, computed only when the needed annotation exists.
Answer: exact match + token-F1 (free-form). ROUGE/BLEU are NOT computed (not_computed) - they are only
meaningful against expert free-text answers, which the baseline datasets do not provide.
Localisation: IoU, precision/recall @ IoU thresholds (single enclosing box per image -> AP not defined here)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence


def normalize_text(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return " ".join(s.split())


def exact_match(pred: str, gt: str) -> float:
    return float(normalize_text(pred) == normalize_text(gt))


def token_f1(pred: str, gt: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gt).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def answer_metrics(preds: List[str], gts: List[str]) -> Dict[str, float]:
    n = len(gts)
    if n == 0:
        return {"n": 0}
    return {"n": n, "exact_match": sum(exact_match(p, g) for p, g in zip(preds, gts)) / n,
            "token_f1": sum(token_f1(p, g) for p, g in zip(preds, gts)) / n,
            "rouge": "not_computed", "bleu": "not_computed"}


def localization_metrics(pred_boxes: List[Sequence[float]], gt_boxes: List[Sequence[float]],
                         thresholds=(0.3, 0.5)) -> Dict[str, float]:
    if not gt_boxes:
        return {"n": 0, "status": "blocked_by_missing_data"}
    ious = [box_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    out = {"n": len(ious), "mean_iou": sum(ious) / len(ious)}
    for t in thresholds:           # one prediction per annotated image -> precision == recall == hit rate
        out[f"hit_rate@{t}"] = sum(i >= t for i in ious) / len(ious)
    return out
