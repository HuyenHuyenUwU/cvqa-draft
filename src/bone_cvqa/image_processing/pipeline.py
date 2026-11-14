"""Khối 1 — Image Processing (dùng chung Offline/Online).

  ảnh -> DICOM Preprocessing -> Vision Encoder -> { Lesion Localization -> ROI Features ; Global Visual Embedding }

Image Quality Assessment (chỉ Online) KHÔNG nằm trong pipeline này; engine Online gọi nó riêng.
`run_until(...)` cho phép chạy độc lập từng bước (dùng bởi run.py).
"""
from typing import Any, Dict

import torch
from torch import nn

from ..commons.types import ImageProcessingOutput, PreprocessedImage
from .dicom_preprocessing import DicomPreprocessor
from .global_embedding.projector import GlobalEmbeddingModule
from .lesion_localization.localization_head import LesionLocalizationHead
from .roi_features.extractor import RoiFeatureExtractor
from .vision_encoder.encoder import VisionEncoder

STAGES = ["dicom_preprocessing", "vision_encoder", "lesion_localization", "roi_features", "global_embedding"]


class ImageProcessingPipeline(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.preprocessor = DicomPreprocessor(config["image_processing"].get("preprocess"))
        self.vision_encoder = VisionEncoder(config["external_models"]["vision_encoder"], d_model)
        self.lesion_localization = LesionLocalizationHead()
        self.roi_features = RoiFeatureExtractor()
        self.global_embedding = GlobalEmbeddingModule(self.vision_encoder.backbone_dim, d_model,
                                                      config["image_processing"].get("global_pooling", "backbone"))

    def preprocess(self, path_or_array) -> PreprocessedImage:
        return self.preprocessor(path_or_array)

    def forward(self, images: torch.Tensor) -> ImageProcessingOutput:
        vision = self.vision_encoder(images)
        localization = self.lesion_localization(vision)
        return ImageProcessingOutput(vision=vision, localization=localization,
                                     roi=self.roi_features(vision, localization),
                                     global_embedding=self.global_embedding(vision))

    def run_until(self, images: torch.Tensor, stage: str) -> Dict[str, Any]:
        """Chạy tới hết `stage` và trả kết quả của các bước đã chạy (để chạy độc lập từng module)."""
        results: Dict[str, Any] = {}
        results["vision_encoder"] = vision = self.vision_encoder(images)
        if stage == "vision_encoder":
            return results
        results["lesion_localization"] = localization = self.lesion_localization(vision)
        if stage == "lesion_localization":
            return results
        results["roi_features"] = self.roi_features(vision, localization)
        if stage == "roi_features":
            return results
        results["global_embedding"] = self.global_embedding(vision)
        return results
