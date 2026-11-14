"""[PLACEHOLDER] LLM qua API trả phí. Không dùng ở giai đoạn này (yêu cầu: chỉ LLM mã nguồn mở chạy local)."""
from typing import Any, Dict

from ..base import BaseModelAdapter
from ..clients.api_client import ApiClient
from ..registry import register_adapter


@register_adapter("medical_llm", "api")
class ApiLLMAdapter(BaseModelAdapter):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.client = ApiClient()

    def load(self) -> None:
        self.client.run()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.generate(inputs)

    def unload(self) -> None:
        self.model = None
