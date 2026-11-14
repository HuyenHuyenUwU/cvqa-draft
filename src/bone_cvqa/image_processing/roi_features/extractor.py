"""ROI Features (khối 1).  in: VisionFeatures + kết quả Lesion Localization  ->  out: RoiFeatures.

Không có box (localization còn là placeholder) -> status=skipped, features=None; Cross-modal Fusion che mask ROI.
`extract_from_box` đã chạy thật khi được cấp box (dùng cho test / khi localization hoàn thiện).
"""
from typing import Optional, Tuple

import torch
from torch import nn

from ...commons.constants import Status
from ...commons.logger import get_logger
from ...commons.types import LocalizationResult, RoiFeatures, VisionFeatures
from .roi_mapper import box_to_patch_mask
from .roi_pooling import masked_mean_pool

log = get_logger("roi_features")


class RoiFeatureExtractor(nn.Module):
    def extract_from_box(self, patch_features: torch.Tensor, grid_size: Tuple[int, int],
                         box_xyxy_norm: torch.Tensor, box_valid: torch.Tensor) -> RoiFeatures:
        mask = box_to_patch_mask(box_xyxy_norm.detach(), grid_size)
        pooled = masked_mean_pool(patch_features, mask) * box_valid.unsqueeze(-1).to(patch_features.dtype)
        return RoiFeatures(status=Status.AVAILABLE, features=pooled, valid=box_valid.clone())

    def forward(self, vision: VisionFeatures, localization: LocalizationResult) -> RoiFeatures:
        if localization.box_xyxy_norm is None:
            log.debug("ROI skipped: localization status=%s (no box)", localization.status)
            return RoiFeatures(status=Status.SKIPPED)
        valid = torch.ones(localization.box_xyxy_norm.shape[0], dtype=torch.bool, device=vision.patch_features.device)
        return self.extract_from_box(vision.patch_features, vision.grid_size, localization.box_xyxy_norm, valid)
