"""Pipeline OFFLINE: dữ liệu -> BoneCVQASystem (dùng chung) -> Trainer.  Module này CHỈ dùng cho huấn luyện."""
from pathlib import Path
from typing import Any, Dict

from ..commons.config import save_config
from ..commons.device import resolve_device
from ..commons.logger import setup_logging
from ..commons.seed import set_seed
from ..data.loaders import make_dataloader
from ..system import build_system
from .trainer import Trainer


class OfflinePipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config["paths"]["output_dir"])
        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        setup_logging(config.get("log_level", "INFO"), str(Path(config["paths"]["log_dir"]) / "training.log"))
        set_seed(int(config.get("seed", 42)))
        self.device = resolve_device(config.get("device", "auto"))

    def make_loader(self, split: str, shuffle: bool):
        return make_dataloader(self.config, split, shuffle, str(self.output_dir / "split_metadata.json"))

    def train(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.config, self.checkpoint_dir / "config.yaml")
        system = build_system(self.config)
        split_info = {"split_metadata_file": str(self.output_dir / "split_metadata.json"),
                      "dataset": self.config["data"]["name"], "split": self.config["data"].get("split")}
        trainer = Trainer(system, self.config, self.make_loader("train", True), self.make_loader("val", False),
                          self.device, split_info, str(self.checkpoint_dir))
        return trainer.fit()
