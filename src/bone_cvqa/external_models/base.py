"""Interface chung cho mọi model ngoài (vision / text / LLM), để thay công nghệ khác khi làm ablation.

    BaseModelAdapter      : interface tối thiểu (load / predict / unload)
    TorchModelAdapter     : cài sẵn phần chung cho model PyTorch (freeze, device, unload)

Adapter là nn.Module nên hệ thống tự thấy tham số của nó (optimizer, state_dict, .to(device)).
Adapter KHÔNG tự bật/tắt gradient: chế độ train/infer do `commons.execution_mode` quyết định.
"""
import gc
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn

from ..commons.device import resolve_device
from ..commons.logger import get_logger


class BaseModelAdapter(nn.Module, ABC):
    role: str = "base"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.model: Any = None
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Tạo/nạp model vào self.model."""

    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Chạy forward. Input/output là dict có khóa được ghi rõ trong docstring từng adapter."""

    @abstractmethod
    def unload(self) -> None:
        """Giải phóng model khỏi bộ nhớ."""

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict(inputs)


class TorchModelAdapter(BaseModelAdapter):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.logger = get_logger(f"external_models.{type(self).__name__}")
        self.trainable = bool(config.get("trainable", False))   # mặc định: ĐÓNG BĂNG model ngoài
        self.device_name = config.get("device", "auto")

    @property
    def device(self) -> torch.device:
        return resolve_device(self.device_name)

    def _finish_load(self) -> None:
        """Gọi ở cuối load(): áp dụng freeze + chuyển device."""
        self.model.to(self.device)
        self.apply_trainable()
        self.is_loaded = True
        self.logger.info("Loaded %s (provider=%s, name=%s, trainable=%s, params=%.1fM)",
                         type(self).__name__, self.config.get("provider"), self.config.get("name"),
                         self.trainable, sum(p.numel() for p in self.model.parameters()) / 1e6)

    def apply_trainable(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = self.trainable
        self.train(self.training)

    def train(self, mode: bool = True):
        # model ngoài bị đóng băng luôn ở eval (không dropout, BatchNorm không cập nhật thống kê)
        return super().train(mode and self.trainable)

    def unload(self) -> None:
        self.model = None
        self.is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
