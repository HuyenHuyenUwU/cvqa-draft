"""Base dataset: records (image-level) -> QA samples with a shared sample interface.

Subclasses only implement ``build_records(cfg)`` (reading THEIR annotation files).
Record schema (all optional except image_id / image_path):
  {image_id, image_path, patient_id, label, label_name, boxes_px: [[x1,y1,x2,y2]], image_size: (w,h)|None, extra}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from ..commons.exceptions import DatasetError
from ..commons.logger import get_logger
from ..commons.constants import ANSWER_SOURCE_WEAK
from .annotations import make_lesion_annotation
from ..image_processing.dicom_preprocessing import DicomPreprocessor
from .qa_generation import load_expert_qa, weak_qa_for_record
from .splits import assign_splits, check_no_leakage, save_split_metadata

log = get_logger("data")


class BaseCVQADataset(Dataset):
    dataset_name = "base"

    def __init__(self, cfg: Dict[str, Any], split: str, preprocessor: DicomPreprocessor,
                 split_metadata_path: Optional[str] = None):
        self.cfg, self.split, self.pre = cfg, split, preprocessor
        records = self.build_records(cfg)
        if not records:
            raise DatasetError(f"No images found for {self.dataset_name} under {cfg.get('root')}")
        for r in records:
            if r.get("image_size") is None:
                try:
                    with Image.open(r["image_path"]) as im:
                        r["image_size"] = im.size
                except Exception:
                    r["image_size"] = None   # DICOM or unreadable: lesion stays 'missing'
        sp = cfg.get("split", {})
        ratios, seed = tuple(sp.get("ratios", (0.7, 0.15, 0.15))), int(sp.get("seed", 42))
        split_of = assign_splits(records, ratios, seed, sp.get("official_split_file"))
        check_no_leakage(records, split_of)
        if split_metadata_path:
            save_split_metadata(split_metadata_path, records, split_of, seed, ratios,
                                "patient_id" if any(r.get("patient_id") for r in records) else "image_id")
        self.split_of = split_of
        self.records = [r for r in records if split_of[str(r["image_id"])] == split]
        self.items = self._build_items(self.records)
        log.info("%s[%s]: %d images -> %d QA samples", self.dataset_name, split, len(self.records), len(self.items))

    # ---- to be provided by subclasses ----
    def build_records(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    # ---- shared ----
    def _build_items(self, records) -> List[Dict[str, Any]]:
        qa_cfg = self.cfg.get("qa", {})
        mode = qa_cfg.get("source", "weak_template")
        expert = load_expert_qa(qa_cfg["file"]) if mode == "file" else {}
        items: List[Dict[str, Any]] = []
        for r in records:
            lesion = make_lesion_annotation(r.get("boxes_px") or [], r.get("image_size"), self.dataset_name)
            turns = expert.get(str(r["image_id"]), []) if mode == "file" else \
                weak_qa_for_record(r, self.dataset_name, lesion["box_xyxy_norm"])
            for k, t in enumerate(turns):
                items.append({"record": r, "lesion": lesion, "turn": k, "history": turns[:k], "qa": t})
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        r, qa = it["record"], it["qa"]
        pre = self.pre(r["image_path"])           # raises InvalidImageError explicitly
        return {
            "sample_id": f"{r['image_id']}__q{it['turn']}",
            "image": pre.tensor,
            "question": qa["question"],
            "conversation_history": [{"question": h["question"], "answer": h["answer"]} for h in it["history"]],
            "answer": qa.get("answer"),
            "lesion_annotation": it["lesion"],
            "metadata": {"dataset": self.dataset_name, "image_id": r["image_id"], "patient_id": r.get("patient_id"),
                         "label": r.get("label"), "label_name": r.get("label_name"), "split": self.split,
                         "answer_source": qa.get("answer_source", ANSWER_SOURCE_WEAK),
                         "original_size_hw": pre.original_size_hw, "extra": r.get("extra", {})},
        }

    @staticmethod
    def scan_images(folder: Path, exts) -> List[Path]:
        exts = {e.lower() for e in exts}
        return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in exts)
