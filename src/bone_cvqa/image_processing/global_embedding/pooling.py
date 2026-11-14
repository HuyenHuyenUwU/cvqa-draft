"""Pooling toàn cục. "backbone": dùng global_feature của backbone (chiều backbone). "mean": trung bình patch (chiều d_model)."""
import torch

from ...commons.types import VisionFeatures


def backbone_pooling(vision: VisionFeatures) -> torch.Tensor:
    return vision.global_feature


def mean_patch_pooling(vision: VisionFeatures) -> torch.Tensor:
    return vision.patch_features.mean(dim=1)
