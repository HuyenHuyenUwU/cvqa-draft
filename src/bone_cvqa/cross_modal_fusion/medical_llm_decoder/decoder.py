"""Medical LLM Decoder (khối 4): nhận soft-prefix đa phương thức + prompt văn bản -> hidden states (teacher forcing)
hoặc câu trả lời sinh tự hồi quy. LLM là model ngoài (adapter, mặc định đóng băng; gradient vẫn chảy xuyên qua).

Chuỗi đưa vào LLM qua `inputs_embeds`:  [soft prefix (chỉ token hợp lệ)] [prompt] [đáp án + EOS]
`labels` = IGNORE_INDEX ở mọi vị trí trừ token đáp án.
"""
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ...commons.constants import IGNORE_INDEX
from ...commons.types import DecoderOutput, FusionOutput
from ...external_models import build_adapter
from .llm_projection import FusionToLLMProjection


class MedicalLLMDecoder(nn.Module):
    def __init__(self, external_config: Dict[str, Any], d_model: int, max_prompt_tokens: int = 192,
                 max_answer_tokens: int = 48) -> None:
        super().__init__()
        self.llm = build_adapter("medical_llm", external_config)
        self.projection = FusionToLLMProjection(d_model, self.llm.hidden_size)
        self.max_prompt_tokens, self.max_answer_tokens = max_prompt_tokens, max_answer_tokens

    # ---- ghép chuỗi ----
    def _token_ids(self, text: str, max_len: int, keep_tail: bool) -> List[int]:
        ids = list(self.llm.tokenizer(text, add_special_tokens=False)["input_ids"])
        return (ids[-max_len:] if keep_tail else ids[:max_len]) if len(ids) > max_len else ids

    def _embed(self, ids: List[int]) -> torch.Tensor:
        return self.llm.predict({"mode": "embed", "input_ids": ids})["embeddings"]

    def _assemble(self, fusion: FusionOutput, prompts: List[str], answers: Optional[List[Optional[str]]], left_pad: bool):
        prefix = self.projection(fusion.tokens)
        device, dtype = prefix.device, self.llm.model.get_input_embeddings().weight.dtype
        sequences, label_rows = [], []
        for i, prompt in enumerate(prompts):
            prefix_i = prefix[i][fusion.mask[i]].to(dtype)
            prompt_embeds = self._embed(self._token_ids(prompt, self.max_prompt_tokens, keep_tail=True)).to(device)
            parts = [prefix_i, prompt_embeds]
            labels = [IGNORE_INDEX] * (prefix_i.shape[0] + prompt_embeds.shape[0])
            if answers is not None and answers[i] is not None:
                answer_ids = self._token_ids(answers[i], self.max_answer_tokens - 1, keep_tail=False) + [self.llm.eos_token_id]
                parts.append(self._embed(answer_ids).to(device))
                labels += answer_ids
            sequences.append(torch.cat(parts, dim=0))
            label_rows.append(labels)
        length, hidden = max(s.shape[0] for s in sequences), prefix.shape[-1]
        embeds = torch.zeros(len(sequences), length, hidden, dtype=dtype, device=device)
        attention = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
        labels = torch.full((len(sequences), length), IGNORE_INDEX, dtype=torch.long, device=device)
        for i, (seq, lab) in enumerate(zip(sequences, label_rows)):
            n = seq.shape[0]
            span = slice(length - n, length) if left_pad else slice(0, n)
            embeds[i, span], attention[i, span] = seq, 1
            labels[i, span] = torch.tensor(lab, device=device)
        return embeds, attention, labels

    def teacher_forcing(self, fusion: FusionOutput, prompts: List[str], answers: List[Optional[str]]) -> DecoderOutput:
        embeds, attention, labels = self._assemble(fusion, prompts, answers, left_pad=False)
        hidden = self.llm.predict({"mode": "hidden", "inputs_embeds": embeds, "attention_mask": attention})["hidden"]
        return DecoderOutput(hidden=hidden, labels=labels, attention_mask=attention)

    def generate(self, fusion: FusionOutput, prompts: List[str], generation: Dict[str, Any]) -> List[str]:
        embeds, attention, _ = self._assemble(fusion, prompts, None, left_pad=True)
        return self.llm.predict({"mode": "generate", "inputs_embeds": embeds, "attention_mask": attention,
                                 **generation})["texts"]
