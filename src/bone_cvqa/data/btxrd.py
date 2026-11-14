"""BTXRD (bone tumor X-ray) adapter.

The exact schema is NOT assumed. Column names, image folder and annotation folder are taken
from configs/data/btxrd.yaml and MUST be set after running scripts/inspect_dataset.py.

Supported (configurable) layout:
  <root>/<images_dir>/*.jpg
  <root>/<metadata_csv>               one row per image (export .xlsx to .csv first)
  <root>/<annotation_dir>/<id>.json   LabelMe-style: {"shapes":[{"points":[[x,y],...]}], "imageWidth","imageHeight"}
Patient identifier column is optional; when present the split is patient-level.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from ..commons.exceptions import DatasetError
from ..commons.logger import get_logger
from .annotations import points_to_xyxy
from .base_dataset import BaseCVQADataset

log = get_logger("data.btxrd")


class BTXRDDataset(BaseCVQADataset):
    dataset_name = "btxrd"

    def build_records(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        root = Path(cfg["root"])
        if not root.is_dir():
            raise DatasetError(f"BTXRD root not found: {root} (set data.root in YAML)")
        cols = cfg.get("columns", {})
        c_id, c_lab, c_pat = cols.get("image_id", "image_id"), cols.get("label", "label"), cols.get("patient_id")
        exts = cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".dcm"])
        meta: Dict[str, Dict[str, str]] = {}
        meta_path = root / cfg.get("metadata_csv", "dataset.csv")
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    meta[Path(str(row[c_id])).stem] = row
        else:
            log.warning("Metadata csv not found at %s -> labels will be missing", meta_path)
        ann_dir = root / cfg.get("annotation_dir", "Annotations")
        label_names = cfg.get("label_names", {})
        recs: List[Dict[str, Any]] = []
        for p in self.scan_images(root / cfg.get("images_dir", "images"), exts):
            row = meta.get(p.stem, {})
            raw = row.get(c_lab)
            boxes, size = [], None
            aj = ann_dir / f"{p.stem}.json"
            if aj.is_file():
                with open(aj, "r", encoding="utf-8") as f:
                    j = json.load(f)
                boxes = [points_to_xyxy(s["points"]) for s in j.get("shapes", []) if s.get("points")]
                if j.get("imageWidth") and j.get("imageHeight"):
                    size = (j["imageWidth"], j["imageHeight"])
            recs.append({
                "image_id": p.stem, "image_path": str(p),
                "patient_id": row.get(c_pat) if c_pat else None,
                "label": raw, "label_name": label_names.get(raw, raw) if raw is not None else None,
                "boxes_px": boxes, "image_size": size, "extra": {"has_metadata": bool(row)},
            })
        return recs
