"""Joint Loss (khối 6 Offline):  L = w_ans*L_ans + w_loc*L_loc + w_rat*L_rat + w_kg*L_kg + w_align*L_align

CHỈ term nào có đủ output của model VÀ nhãn thật mới ACTIVE:
  answer           : cross-entropy trên token đáp án (cần đáp án)            -> ACTIVE
  localization/rationale/knowledge_graph/alignment                           -> SKIPPED (placeholder), kèm lý do
Term bị bỏ không bao giờ được thay bằng hằng số. Chỉ Offline dùng module này.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..commons.logger import get_logger

log = get_logger("training.joint_loss")

SKIP_REASONS = {
    "localization": "not_implemented: Lesion Localization is a placeholder",
    "rationale": "not_implemented: no ground-truth explanation / explanation head",
    "knowledge_graph": "not_implemented: ontology branch and Evidence Graph are placeholders",
    "alignment": "not_implemented: needs Evidence Graph + explanation module",
}


@dataclass
class JointLossResult:
    total: Optional[torch.Tensor]                 # None nếu không có term nào active
    components: Dict[str, float] = field(default_factory=dict)
    active: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)


class JointLoss:
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = {"answer": 1.0, **{name: 0.0 for name in SKIP_REASONS}, **(weights or {})}
        self._warned = False

    def __call__(self, answer_logits: Optional[torch.Tensor], answer_targets: Optional[torch.Tensor]) -> JointLossResult:
        skipped = dict(SKIP_REASONS)
        terms: Dict[str, torch.Tensor] = {}
        if answer_logits is not None and answer_targets is not None and answer_targets.numel() > 0:
            terms["answer"] = F.cross_entropy(answer_logits.float(), answer_targets)
        else:
            skipped["answer"] = "no valid answer supervision in this batch"
        active = [name for name in terms if self.weights.get(name, 0.0) > 0]
        for name in terms:
            if name not in active:
                skipped[name] = "weight is 0"
        if not self._warned:
            log.warning("Multi-task objective NOT fully trained. Active: %s. Skipped: %s", active, sorted(skipped))
            self._warned = True
        total = sum(self.weights[name] * terms[name] for name in active) if active else None
        return JointLossResult(total=total, components={n: float(terms[n].detach()) for n in active},
                               active=active, skipped=skipped)
