"""Answer Head = đầu LM (lm_head) của LLM: hidden states -> logits token (không dùng tập đáp án cố định).
Chỉ tính logits tại vị trí đáp án (tiết kiệm bộ nhớ vì vocab lớn)."""
from typing import Tuple

import torch

from ...commons.constants import IGNORE_INDEX
from ...commons.types import DecoderOutput
from ...external_models.base import BaseModelAdapter


class AnswerHead:
    def __init__(self, llm: BaseModelAdapter) -> None:
        self.llm = llm

    def logits_at_answer_positions(self, decoder_output: DecoderOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (logits [M, V], targets [M]) với M = số token đáp án (đã dịch 1 vị trí cho dự đoán token kế tiếp)."""
        labels = decoder_output.labels[:, 1:]
        selected = labels != IGNORE_INDEX
        hidden = decoder_output.hidden[:, :-1][selected]
        return self.llm.predict({"mode": "lm_head", "hidden": hidden})["logits"], labels[selected]
