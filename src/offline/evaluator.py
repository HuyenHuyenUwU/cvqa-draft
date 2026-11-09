"""Offline evaluation on a FIXED split. Metrics are computed only where annotations exist.
Image-quality distribution is reported by the Online evaluation (it owns that module);
ontology / uncertainty are reported as not_implemented - no fabricated numbers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..models.cvqa_inputs import build_model_inputs
from ..utils.metrics import answer_metrics, localization_metrics


@torch.no_grad()
def evaluate(model, loader, qp, ontology, device, gen_cfg: Optional[Dict[str, Any]] = None,
             out_dir: Optional[str] = None, split_name: str = "test", max_batches: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    gen_cfg = gen_cfg or {}
    rows, preds, gts, pbox, gbox = [], [], [], [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        inp = build_model_inputs(model, qp, ontology, batch["question"], batch["conversation_history"])
        out = model.generate(batch["image"].to(device), inp["conv_texts"], inp["type_ids"], inp["prompts"],
                             knowledge=inp["knowledge"], max_new_tokens=gen_cfg.get("max_new_tokens"),
                             do_sample=bool(gen_cfg.get("do_sample", False)))
        for i, sid in enumerate(batch["sample_id"]):
            gt = batch["answer"][i]
            rows.append({"sample_id": sid, "question": batch["question"][i], "prediction": out["texts"][i],
                         "ground_truth": gt, "answer_source": batch["metadata"][i].get("answer_source")})
            if gt is not None:
                preds.append(out["texts"][i]); gts.append(gt)
            if out["loc"] is not None and model.localizer_trained and bool(batch["has_box"][i]):
                pbox.append(out["loc"]["box_xyxy_norm"][i].tolist()); gbox.append(batch["gt_box"][i].tolist())
    res = {"split": split_name, "num_samples": len(rows),
           "answer": answer_metrics(preds, gts),
           "answer_supervision_note": "answers may be template weak-supervision, not expert ground truth",
           "localization": localization_metrics(pbox, gbox) if model.localizer_trained
           else {"status": "not_available", "reason": "localizer not trained"},
           "ontology": {"status": "not_implemented"}, "uncertainty": {"status": "not_implemented"}}
    if out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        (d / f"eval_{split_name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        if rows:
            with open(d / f"predictions_{split_name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
    return res
