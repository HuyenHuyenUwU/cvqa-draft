"""[PLACEHOLDER] Client gọi model qua dịch vụ nội bộ (HTTP/gRPC)."""
from typing import Any, Dict

from ...commons.placeholder import PlaceholderModule


class ServiceClient(PlaceholderModule):
    module_name = "Service client for remote model server"

    def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(payload)
