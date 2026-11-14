"""Sinh câu trả lời tự hồi quy (dùng ở Online / Evaluate)."""
from typing import Any, Dict, List

from ...commons.types import FusionOutput


class AnswerGenerator:
    def __init__(self, decoder) -> None:
        self.decoder = decoder

    def generate(self, fusion: FusionOutput, prompts: List[str], generation: Dict[str, Any]) -> List[str]:
        return self.decoder.generate(fusion, prompts, generation)
