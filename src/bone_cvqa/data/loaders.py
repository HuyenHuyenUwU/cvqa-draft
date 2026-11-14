"""DataLoader dùng chung (Offline huấn luyện và Evaluate)."""
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from ..image_processing.dicom_preprocessing import DicomPreprocessor
from .collate import cvqa_collate
from .dataset_factory import build_dataset


def make_dataloader(config: Dict[str, Any], split: str, shuffle: bool, split_metadata_path: Optional[str] = None) -> DataLoader:
    preprocessor = DicomPreprocessor(config["image_processing"]["preprocess"])
    dataset = build_dataset(config, split, preprocessor, split_metadata_path)
    train_cfg = config.get("train", {})
    return DataLoader(dataset, batch_size=int(train_cfg.get("batch_size", 2)), shuffle=shuffle,
                      num_workers=int(train_cfg.get("num_workers", 0)), collate_fn=cvqa_collate)
