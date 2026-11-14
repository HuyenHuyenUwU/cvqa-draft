"""Tạo dataset theo cfg["data"]: name + datasets.<name> (+ split/qa dùng chung)."""
from typing import Any, Dict, Optional

from ..commons.exceptions import DatasetError
from ..image_processing.dicom_preprocessing import DicomPreprocessor
from .btxrd import BTXRDDataset
from .fracatlas import FracAtlasDataset

REGISTRY = {"fracatlas": FracAtlasDataset, "btxrd": BTXRDDataset}


def build_dataset(config: Dict[str, Any], split: str, preprocessor: DicomPreprocessor,
                  split_metadata_path: Optional[str] = None):
    data_cfg = config["data"]
    name = str(data_cfg.get("name", "")).lower()
    if name not in REGISTRY:
        raise DatasetError(f"Dataset không hỗ trợ '{name}'. Có: {sorted(REGISTRY)}")
    dataset_cfg = {"name": name, "split": data_cfg.get("split", {}), "qa": data_cfg.get("qa", {}),
                   **data_cfg["datasets"][name]}
    return REGISTRY[name](dataset_cfg, split, preprocessor, split_metadata_path)
