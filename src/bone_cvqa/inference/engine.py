"""InferenceEngine: nạp checkpoint (checkpoint_io dùng chung) -> OnlinePipeline.
Không dùng Trainer / Loss / Optimizer; cấu hình tiền xử lý lấy từ checkpoint nên giống hệt lúc train."""
from typing import Any, Dict, List, Optional

from ..checkpoint_io import load_system_from_checkpoint
from ..commons.logger import setup_logging
from .pipeline import OnlinePipeline


class InferenceEngine:
    def __init__(self, pipeline: OnlinePipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: Dict[str, Any]) -> "InferenceEngine":
        """`config` (base+model+inference) cung cấp phần inference và đường dẫn model ngoài hợp lệ trên máy này."""
        setup_logging(config.get("log_level", "INFO"))
        system, checkpoint = load_system_from_checkpoint(checkpoint_path, {"external_models": config.get("external_models", {})})
        merged = {**checkpoint["model_config"], "inference": config["inference"]}
        return cls(OnlinePipeline(system.eval(), merged))

    def infer(self, image_path: str, question: str, history: Optional[List[Any]] = None, sample_id: str = "sample") -> Dict[str, Any]:
        return self.pipeline.run(image_path, question, history, sample_id)
