"""Offline orchestration (diagram: pipeline_offline.png).
Image Processing -> Question Processing -> Ontology Processing (placeholders) -> Cross-modal Fusion ->
Answer Generation -> Multi-task Learning. Does not depend on anything in ``src.online``."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from ..common.config import save_config
from ..common.device import resolve_device
from ..common.logger import get_logger, setup_logging
from ..common.seed import set_seed
from ..data.collate import cvqa_collate
from ..data.dataset_factory import build_dataset
from ..data.image_preprocessing import ImagePreprocessor
from ..models.model_io import build_cvqa_model, load_model_checkpoint
from ..ontology.branch import OntologyBranch
from ..question_processing.processor import QuestionProcessor
from .evaluator import evaluate
from .trainer import Trainer

log = get_logger("offline.pipeline")


class OfflinePipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        out = cfg.get("output", {})
        self.out_dir = Path(out.get("output_dir", "outputs/exp"))
        self.ckpt_dir = Path(out.get("checkpoint_dir", "checkpoints/exp"))
        setup_logging(cfg.get("log_level", "INFO"), str(Path(out.get("log_dir", "logs")) / "offline.log"))
        set_seed(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", False)))
        self.device = resolve_device(cfg.get("device", "auto"))
        self.pre = ImagePreprocessor(cfg["preprocess"])
        self.qp, self.onto = QuestionProcessor(), OntologyBranch()

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        tc = self.cfg.get("train", {})
        ds = build_dataset(self.cfg["data"], split, self.pre, str(self.out_dir / "split_metadata.json"))
        return DataLoader(ds, batch_size=int(tc.get("batch_size", 2)), shuffle=shuffle,
                          num_workers=int(tc.get("num_workers", 0)), collate_fn=cvqa_collate)

    def train(self) -> Dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, self.ckpt_dir / "config.yaml")          # config saved next to checkpoints
        tl, vl = self._loader("train", True), self._loader("val", False)
        model = build_cvqa_model(self.cfg)
        split_info = {"split_metadata_file": str(self.out_dir / "split_metadata.json"), "dataset": self.cfg["data"].get("name"),
                      "split_cfg": self.cfg["data"].get("split")}
        tr = Trainer(model, self.cfg, tl, vl, self.qp, self.onto, self.device, split_info, str(self.ckpt_dir))
        return tr.fit()

    def evaluate(self, checkpoint: str, split: str = "test") -> Dict[str, Any]:
        model, _ = load_model_checkpoint(checkpoint, cfg_override={k: self.cfg[k] for k in ("biomedclip", "llm")
                                                                   if k in self.cfg})
        model.to(self.device)
        return evaluate(model, self._loader(split, False), self.qp, self.onto, self.device,
                        self.cfg.get("generation"), str(self.out_dir), split)
