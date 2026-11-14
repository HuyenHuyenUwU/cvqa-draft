"""Đánh giá trên một split CỐ ĐỊNH. Chạy ở chế độ INFER (không gradient). Lưu JSON + CSV dự đoán.
Ontology / uncertainty / localization: not_implemented (không báo số bịa)."""
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..commons.constants import ExecutionMode
from ..commons.execution_mode import execution_context
from ..system import BoneCVQASystem
from .metrics import answer_metrics


def evaluate(system: BoneCVQASystem, loader, device, generation: Dict[str, Any], out_dir: Optional[str] = None,
             split_name: str = "test", max_batches: Optional[int] = None) -> Dict[str, Any]:
    rows, predictions, references = [], [], []
    with execution_context(system, ExecutionMode.INFER):
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            context = system.encode(batch["image"].to(device), batch["question"], batch["conversation_history"])
            texts = system.generate_answers(context, generation)
            for i, text in enumerate(texts):
                reference = batch["answer"][i]
                rows.append({"sample_id": batch["sample_id"][i], "question": batch["question"][i], "prediction": text,
                             "reference": reference, "answer_source": batch["metadata"][i].get("answer_source")})
                if reference is not None:
                    predictions.append(text); references.append(reference)
    result = {"split": split_name, "num_samples": len(rows), "answer": answer_metrics(predictions, references),
              "note": "reference answers may be template weak-supervision, not expert ground truth",
              "localization": {"status": "not_implemented"}, "ontology": {"status": "not_implemented"},
              "uncertainty": {"status": "not_implemented"}}
    if out_dir:
        folder = Path(out_dir); folder.mkdir(parents=True, exist_ok=True)
        (folder / f"eval_{split_name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        if rows:
            with open(folder / f"predictions_{split_name}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    return result
