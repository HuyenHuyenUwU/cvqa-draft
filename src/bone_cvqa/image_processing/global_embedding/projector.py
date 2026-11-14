"""Global Visual Embedding (khối 1): VisionFeatures -> vector toàn cục [B, d_model]. Độc lập với Lesion Localization."""
from torch import nn

from ...commons.types import GlobalEmbedding, VisionFeatures
from ..vision_encoder.projection import FeatureProjection
from .pooling import backbone_pooling, mean_patch_pooling


class GlobalEmbeddingModule(nn.Module):
    def __init__(self, backbone_dim: int, d_model: int, pooling: str = "backbone") -> None:
        super().__init__()
        assert pooling in ("backbone", "mean")
        self.pooling = pooling
        self.projection = FeatureProjection(backbone_dim if pooling == "backbone" else d_model, d_model)

    def forward(self, vision: VisionFeatures) -> GlobalEmbedding:
        pooled = backbone_pooling(vision) if self.pooling == "backbone" else mean_patch_pooling(vision)
        return GlobalEmbedding(embedding=self.projection(pooled))
