"""Cross-modal Fusion.

Every modality is projected to ``d_model`` and becomes one (or T) token(s) with a learned
modality embedding; a small Transformer encoder mixes them (self-attention over the concatenated
token set -> cross-modal interaction). Cheap, handles variable token counts via key-padding masks,
and can be swapped for a dedicated cross-attention module by replacing ``self.mixer``.

Slots:  [global][roi][question_emb][type][knowledge] + question tokens (+ optional patch tokens)
Missing modalities (ROI without box, knowledge branch not implemented) are *masked out*, never
filled with fabricated values. The Type-Conditional Prompt enters as a learned embedding of its
question-type id (the prompt TEXT additionally reaches the LLM through its text prompt).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...common.logger import get_logger
from .projection import ModalityProjection

log = get_logger("models.fusion")
MODALITIES = ["global", "roi", "question", "type", "knowledge", "question_tokens", "patches"]


@dataclass
class FusionInputs:
    global_embedding: torch.Tensor                        # [B, Dg]
    question_embedding: torch.Tensor                      # [B, De]
    question_tokens: torch.Tensor                         # [B, T, Dt]
    question_mask: torch.Tensor                           # [B, T] bool
    type_id: torch.Tensor                                 # [B] long
    roi_features: Optional[torch.Tensor] = None           # [B, Dr]
    roi_valid: Optional[torch.Tensor] = None              # [B] bool
    knowledge_embedding: Optional[torch.Tensor] = None    # [B, Dk] or None (ontology unavailable)
    knowledge_status: str = "not_implemented"
    patch_features: Optional[torch.Tensor] = None         # [B, N, Dv]


@dataclass
class FusionOutput:
    tokens: torch.Tensor                                  # [B, K, d_model]
    mask: torch.Tensor                                    # [B, K] bool (True = valid)
    modality_status: Dict[str, str]


class MultimodalFusion(nn.Module):
    def __init__(self, dims: Dict[str, int], d_model: int = 256, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, num_question_types: int = 6, use_patch_tokens: bool = False,
                 max_question_tokens: int = 512):
        super().__init__()
        self.d_model, self.use_patch_tokens = d_model, use_patch_tokens
        self.proj = nn.ModuleDict({
            "global": ModalityProjection(dims["global"], d_model),
            "roi": ModalityProjection(dims["roi"], d_model),
            "question": ModalityProjection(dims["question"], d_model),
            "question_tokens": ModalityProjection(dims["question_tokens"], d_model),
        })
        if dims.get("knowledge"):
            self.proj["knowledge"] = ModalityProjection(dims["knowledge"], d_model)
        if use_patch_tokens:
            self.proj["patches"] = ModalityProjection(dims["patches"], d_model)
        self.type_emb = nn.Embedding(num_question_types, d_model)
        self.modality_emb = nn.Embedding(len(MODALITIES), d_model)
        self.pos_emb = nn.Embedding(max_question_tokens, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self._logged_knowledge_skip = False

    def _tag(self, x, name):
        return x + self.modality_emb.weight[MODALITIES.index(name)]

    def forward(self, f: FusionInputs) -> FusionOutput:
        B, dev = f.global_embedding.shape[0], f.global_embedding.device
        toks, masks, status = [], [], {}
        ones = torch.ones(B, 1, dtype=torch.bool, device=dev)

        toks.append(self._tag(self.proj["global"](f.global_embedding), "global").unsqueeze(1)); masks.append(ones)
        status["global"] = "available"

        if f.roi_features is not None and f.roi_valid is not None:
            toks.append(self._tag(self.proj["roi"](f.roi_features), "roi").unsqueeze(1))
            masks.append(f.roi_valid.unsqueeze(1))
            status["roi"] = "available" if bool(f.roi_valid.any()) else "missing"
        else:
            toks.append(torch.zeros(B, 1, self.d_model, device=dev)); masks.append(~ones)
            status["roi"] = "missing"

        toks.append(self._tag(self.proj["question"](f.question_embedding), "question").unsqueeze(1)); masks.append(ones)
        toks.append(self._tag(self.type_emb(f.type_id), "type").unsqueeze(1)); masks.append(ones)
        status["question"] = status["type"] = "available"

        if f.knowledge_embedding is not None and "knowledge" in self.proj and f.knowledge_status == "available":
            toks.append(self._tag(self.proj["knowledge"](f.knowledge_embedding), "knowledge").unsqueeze(1))
            masks.append(ones)
            status["knowledge"] = "available"
        else:
            toks.append(torch.zeros(B, 1, self.d_model, device=dev)); masks.append(~ones)
            status["knowledge"] = f.knowledge_status if f.knowledge_status != "available" else "missing"
            if not self._logged_knowledge_skip:
                log.warning("Ontology/knowledge input skipped in fusion (status=%s); masked out, nothing fabricated.",
                            status["knowledge"])
                self._logged_knowledge_skip = True

        T = f.question_tokens.shape[1]
        qt = self.proj["question_tokens"](f.question_tokens) + self.pos_emb(torch.arange(T, device=dev))
        toks.append(self._tag(qt, "question_tokens")); masks.append(f.question_mask)
        status["question_tokens"] = "available"

        if self.use_patch_tokens and f.patch_features is not None:
            toks.append(self._tag(self.proj["patches"](f.patch_features), "patches"))
            masks.append(torch.ones(B, f.patch_features.shape[1], dtype=torch.bool, device=dev))
            status["patches"] = "available"

        x, m = torch.cat(toks, 1), torch.cat(masks, 1)
        y = self.mixer(x, src_key_padding_mask=~m)
        y = self.out_norm(y) * m.unsqueeze(-1).to(y.dtype)
        return FusionOutput(tokens=y, mask=m, modality_status=status)
