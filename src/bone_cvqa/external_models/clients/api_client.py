"""[PLACEHOLDER] Client gọi LLM qua API (có phí). Dự án KHÔNG dùng API trả phí ở giai đoạn này."""
from typing import Any, Dict

from ...commons.placeholder import PlaceholderModule


class ApiClient(PlaceholderModule):
    module_name = "API client for hosted LLM"

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(payload)
