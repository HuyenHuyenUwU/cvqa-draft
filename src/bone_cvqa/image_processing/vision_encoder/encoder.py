"""Vision Encoder (khối 1): ảnh -> VisionFeatures.  patch_features đã project về d_model; global_feature giữ chiều backbone."""
from typing import Any, Dict

import torch
from torch import nn

from ...commons.types import VisionFeatures
from .backbone import build_vision_backbone
from .feature_extractor import VisionFeatureExtractor
from .projection import FeatureProjection


class VisionEncoder(nn.Module):
    def __init__(self, external_config: Dict[str, Any], d_model: int) -> None:
        super().__init__()
        self.backbone = build_vision_backbone(external_config)
        self.extractor = VisionFeatureExtractor(self.backbone)
        self.backbone_dim = self._probe_dim()
        self.patch_projection = FeatureProjection(self.backbone_dim, d_model)

    def _probe_dim(self) -> int:
        with torch.inference_mode():
            probe = self.extractor.extract(torch.zeros(1, 3, 64, 64))
        return int(probe["patch_features"].shape[-1])

    def forward(self, images: torch.Tensor) -> VisionFeatures:
        raw = self.extractor.extract(images)
        return VisionFeatures(patch_features=self.patch_projection(raw["patch_features"]),
                              global_feature=raw["global_feature"], grid_size=raw["grid_size"])
