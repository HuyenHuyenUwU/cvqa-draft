"""Inference engine: loads an inference checkpoint and exposes ``infer``. Independent of the Offline trainer
(uses only shared ``models.model_io``). The preprocessing config is taken from the checkpoint so that training
and inference use IDENTICAL preprocessing."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..common.device import resolve_device
from ..common.logger import get_logger, setup_logging
from ..data.image_preprocessing import ImagePreprocessor
from ..models.model_io import load_model_checkpoint
from .pipeline import OnlinePipeline

log = get_logger("online.engine")


class InferenceEngine:
    def __init__(self, pipeline: OnlinePipeline):
        self.pipeline = pipeline

    @classmethod
    def from_checkpoint(cls, checkpoint: str, online_cfg: Optional[Dict[str, Any]] = None) -> "InferenceEngine":
        c = online_cfg or {}
        setup_logging(c.get("log_level", "INFO"), c.get("log_file"))
        override = {k: c[k] for k in ("biomedclip", "llm", "text_encoder") if k in c}   # machine-local paths
        model, ck = load_model_checkpoint(checkpoint, cfg_override=override)
        pre = ImagePreprocessor(ck["model_config"]["preprocess"])
        return cls(OnlinePipeline(model, pre, resolve_device(c.get("device", "auto")), c))

    def infer(self, image_path: str, question: str, history: Optional[List[Any]] = None,
              sample_id: str = "sample") -> Dict[str, Any]:
        return self.pipeline.run(image_path, question, history, sample_id)
