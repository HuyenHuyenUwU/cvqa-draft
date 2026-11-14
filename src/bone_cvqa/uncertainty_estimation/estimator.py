"""[PLACEHOLDER] Uncertainty Estimation và Confidence Score — CHỈ CÓ Ở ONLINE (không có trong sơ đồ Offline).
Trả score=None, status=not_implemented. Không dùng xác suất token thô như độ tin cậy y khoa đã hiệu chuẩn."""
from typing import Any, Dict

from ..commons.constants import Status
from ..commons.placeholder import PlaceholderModule

ONLINE_ONLY = True


class UncertaintyEstimator(PlaceholderModule):
    module_name = "Uncertainty Estimation"

    def empty_result(self) -> Dict[str, Any]:
        return {"score": None, "status": Status.NOT_IMPLEMENTED}

    def estimate(self, logits=None, multimodal_representation=None, answer=None) -> Dict[str, Any]:
        return self.run(logits, multimodal_representation, answer)


class ConfidenceScorer(PlaceholderModule):
    module_name = "Confidence Score"

    def empty_result(self) -> Dict[str, Any]:
        return {"score": None, "status": Status.NOT_IMPLEMENTED}

    def score(self, uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(uncertainty)
