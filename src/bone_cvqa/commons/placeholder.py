"""Lớp cơ sở cho mọi module placeholder (ontology, localization, uncertainty, ...).

Quy ước:
  * Lần gọi đầu tiên ghi log WARNING: "[TODO] <tên module> is a placeholder. ..."
  * Các lần sau ghi DEBUG (tránh spam log khi train).
  * Kết quả luôn có `status = not_implemented`; không bao giờ bịa số liệu/embedding.
"""
from typing import Any, Dict

from .constants import Status
from .logger import get_logger, placeholder_message


class PlaceholderModule:
    module_name: str = "Unnamed module"

    def __init__(self) -> None:
        self._logger = get_logger(f"placeholder.{type(self).__name__}")
        self._warned = False

    def empty_result(self) -> Dict[str, Any]:
        """Kết quả rỗng có tài liệu; lớp con ghi đè để thêm các trường riêng."""
        return {"status": Status.NOT_IMPLEMENTED}

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        message = placeholder_message(self.module_name)
        if not self._warned:
            self._logger.warning(message)
            self._warned = True
        else:
            self._logger.debug(message)
        return self.empty_result()

    __call__ = run
