"""[PLACEHOLDER] Heatmap / Bounding Box (đầu ra khối 5). Phụ thuộc Lesion Localization chưa có."""
from typing import Any, Dict

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule


class HeatmapBoundingBoxPredictor(PlaceholderModule):
    module_name = "Heatmap / Bounding Box output"

    def empty_result(self) -> Dict[str, Any]:
        return {"type": None, "value": None, "status": Status.NOT_IMPLEMENTED}
