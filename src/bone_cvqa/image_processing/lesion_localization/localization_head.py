"""[PLACEHOLDER] Lesion Localization.  in: VisionFeatures  ->  out: box [B,4] xyxy chuẩn hóa và/hoặc heatmap [B,N].
Hiện chỉ trả status=not_implemented, không bịa box/heatmap."""
from torch import nn

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule
from ...commons.types import LocalizationResult, VisionFeatures


class LesionLocalizationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._placeholder = PlaceholderModule()
        self._placeholder.module_name = "Lesion Localization"

    def forward(self, vision: VisionFeatures) -> LocalizationResult:
        self._placeholder.run(vision)
        return LocalizationResult(status=Status.NOT_IMPLEMENTED)
