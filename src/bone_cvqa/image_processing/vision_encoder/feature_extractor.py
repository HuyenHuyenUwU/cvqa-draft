"""Gọi backbone và trả đặc trưng thô (chưa project)."""
from typing import Any, Dict

import torch

from ...external_models.base import BaseModelAdapter


class VisionFeatureExtractor:
    def __init__(self, backbone: BaseModelAdapter) -> None:
        self.backbone = backbone

    def extract(self, images: torch.Tensor) -> Dict[str, Any]:
        """images [B,3,H,W] -> {"patch_features": [B,N,Dv], "global_feature": [B,Dg], "grid_size": (h,w)}"""
        return self.backbone.predict({"images": images})
