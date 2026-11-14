"""[PLACEHOLDER] LLaVA-Med làm decoder (nhận ảnh trực tiếp, khác cơ chế soft-prefix hiện tại)."""
from typing import Any, Dict

from ...commons.placeholder import PlaceholderModule
from ..base import BaseModelAdapter
from ..registry import register_adapter


@register_adapter("medical_llm", "llava_med")
class LlavaMedAdapter(BaseModelAdapter):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._placeholder = PlaceholderModule()
        self._placeholder.module_name = "LLaVA-Med adapter"

    def load(self) -> None:
        self._placeholder.run()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._placeholder.run(inputs)

    def unload(self) -> None:
        self.model = None
