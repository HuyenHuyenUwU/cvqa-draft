"""Joint Loss  L = l_ans*L_ans + l_rat*L_rat + l_loc*L_loc + l_kg*L_kg + l_align*L_align

Only terms backed by BOTH a real model output and real ground truth are ACTIVE:
  L_ans   : token cross-entropy on answer tokens (needs answers)
  L_loc   : L1 + GIoU (+ optional box-induced patch BCE) on rows with an annotated box
  L_rat, L_kg, L_align : SKIPPED (no rationale ground truth / ontology branch / evidence graph yet).
Skipped terms are never replaced by constants and are reported in ``skipped`` with a reason.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..models.localization.lesion_localizer import box_to_patch_mask

SKIP_REASONS = {
    "rat": "not_implemented: no ground-truth rationale and no rationale generation head",
    "kg": "not_implemented: ontology retrieval/evidence graph are placeholders",
    "align": "not_implemented: needs evidence graph + explanation module",
}


def giou_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ix1, iy1 = torch.max(pred[:, 0], gt[:, 0]), torch.max(pred[:, 1], gt[:, 1])
    ix2, iy2 = torch.min(pred[:, 2], gt[:, 2]), torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ap = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = ap + ag - inter + 1e-7
    cx1, cy1 = torch.min(pred[:, 0], gt[:, 0]), torch.min(pred[:, 1], gt[:, 1])
    cx2, cy2 = torch.max(pred[:, 2], gt[:, 2]), torch.max(pred[:, 3], gt[:, 3])
    carea = (cx2 - cx1) * (cy2 - cy1) + 1e-7
    giou = inter / union - (carea - union) / carea
    return (1 - giou).mean()


class JointLoss:
    def __init__(self, weights: Optional[Dict[str, float]] = None, heatmap_bce: bool = True):
        w = {"ans": 1.0, "loc": 1.0, "rat": 0.0, "kg": 0.0, "align": 0.0}
        w.update(weights or {})
        self.w, self.heatmap_bce = w, heatmap_bce

    def __call__(self, out: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        comps: Dict[str, torch.Tensor] = {}
        skipped: Dict[str, str] = {k: v for k, v in SKIP_REASONS.items()}

        if "answer_logits" in out and out["answer_targets"].numel() > 0:
            comps["ans"] = F.cross_entropy(out["answer_logits"].float(), out["answer_targets"])
        else:
            skipped["ans"] = "no valid answer supervision in this batch"

        loc, has_box = out.get("loc"), batch["has_box"].to(out["roi_valid"].device)
        if loc is not None and bool(has_box.any()):
            gt = batch["gt_box"].to(has_box.device)[has_box]
            pb = loc["box_xyxy_norm"][has_box]
            l = F.l1_loss(pb, gt) + giou_loss(pb, gt)
            if self.heatmap_bce:
                tgt = box_to_patch_mask(gt, out["grid_size"])
                l = l + F.binary_cross_entropy_with_logits(loc["heatmap_logits"][has_box], tgt)
            comps["loc"] = l
        else:
            skipped["loc"] = "no lesion annotation in this batch (target marked missing)" if loc is not None \
                else "localizer not available"

        total = sum(self.w[k] * v for k, v in comps.items() if self.w.get(k, 0) > 0) if comps else None
        zero_w = [k for k in comps if self.w.get(k, 0) <= 0]
        for k in zero_w:
            skipped[k] = "weight is 0"
        active = [k for k in comps if self.w.get(k, 0) > 0]
        if not active:
            total = None
        return {"total": total, "components": {k: float(v.detach()) for k, v in comps.items() if k in active},
                "active": active, "skipped": {k: v for k, v in skipped.items() if k not in active}}
