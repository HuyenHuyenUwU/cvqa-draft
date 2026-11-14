"""Backbone = model ngoài (ResNet50 mặc định), tạo qua registry để thay được (BiomedCLIP, ...)."""
from typing import Any, Dict

from ...external_models import build_adapter
from ...external_models.base import BaseModelAdapter


def build_vision_backbone(external_config: Dict[str, Any]) -> BaseModelAdapter:
    return build_adapter("vision_encoder", external_config)
