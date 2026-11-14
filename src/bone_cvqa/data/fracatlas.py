"""FracAtlas adapter.

ASSUMED layout (from public dataset documentation; VERIFY with scripts/inspect_dataset.py):
  <root>/images/Fractured/*.jpg , <root>/images/Non_fractured/*.jpg
  <root>/Annotations/COCO JSON/COCO_fracture_masks.json   (bbox = [x, y, w, h])
No patient identifiers are assumed -> split is by image_id.
All paths / folder names are configurable in configs/data/fracatlas.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..commons.exceptions import DatasetError
from ..commons.logger import get_logger
from .annotations import xywh_to_xyxy
from .base_dataset import BaseCVQADataset

log = get_logger("data.fracatlas")


class FracAtlasDataset(BaseCVQADataset):
    dataset_name = "fracatlas"

    def build_records(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        root = Path(cfg["root"])
        if not root.is_dir():
            raise DatasetError(f"FracAtlas root not found: {root} (set data.root in YAML)")
        exts = cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".dcm"])
        coco = {}
        coco_path = root / cfg.get("coco_json", "Annotations/COCO JSON/COCO_fracture_masks.json")
        if coco_path.is_file():
            with open(coco_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
        else:
            log.warning("COCO json not found at %s -> lesion annotations will be 'missing'", coco_path)
        by_name = {Path(im["file_name"]).name: im for im in coco.get("images", [])}
        boxes_by_id: Dict[Any, list] = {}
        for a in coco.get("annotations", []):
            if a.get("bbox"):
                boxes_by_id.setdefault(a["image_id"], []).append(xywh_to_xyxy(a["bbox"]))

        label_dirs: Dict[str, int] = cfg.get("label_dirs", {"Fractured": 1, "Non_fractured": 0})
        recs: List[Dict[str, Any]] = []
        for sub, lab in label_dirs.items():
            for p in self.scan_images(root / cfg.get("images_dir", "images") / sub, exts):
                info = by_name.get(p.name)
                recs.append({
                    "image_id": p.stem, "image_path": str(p), "patient_id": None,
                    "label": lab, "label_name": "fractured" if lab == 1 else "non_fractured",
                    "boxes_px": boxes_by_id.get(info["id"], []) if info else [],
                    "image_size": (info["width"], info["height"]) if info and "width" in info else None,
                    "extra": {"has_coco_entry": info is not None},
                })
        return recs
