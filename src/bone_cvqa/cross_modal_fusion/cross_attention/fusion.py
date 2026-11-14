"""Cross-modal Fusion (khối 4) bằng CROSS-ATTENTION.

  Query  : token câu hỏi (QuestionFeatures, đã ở d_model)
  Memory : [global embedding] + [ROI] + [type prompt] + [knowledge] + [patch tokens]  (mỗi loại có modality embedding)
  Output : token câu hỏi đã "nhìn" ảnh, shape [B, T, d_model]

Modality thiếu (ROI không có box, knowledge là placeholder) bị che bằng memory mask — KHÔNG điền giá trị bịa.
"""
from typing import Dict, List

import torch
from torch import nn

from ...commons.constants import QuestionType, Status
from ...commons.logger import get_logger
from ...commons.types import FusionOutput, ImageProcessingOutput, OntologyOutput, QuestionFeatures
from .cross_attention_layer import CrossAttentionLayer

log = get_logger("cross_modal_fusion")
MEMORY_KINDS = ["global", "roi", "type", "knowledge", "patch"]


class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1,
                 knowledge_dim: int = None) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(len(QuestionType.ALL), d_model)
        self.kind_embedding = nn.Embedding(len(MEMORY_KINDS), d_model)
        self.knowledge_projection = nn.Linear(knowledge_dim, d_model) if knowledge_dim else None
        self.layers = nn.ModuleList([CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(d_model)
        self._logged_skip = False

    def _tag(self, tokens: torch.Tensor, kind: str) -> torch.Tensor:
        return tokens + self.kind_embedding.weight[MEMORY_KINDS.index(kind)]

    def build_memory(self, image: ImageProcessingOutput, ontology: OntologyOutput, type_ids: torch.Tensor):
        batch, device = type_ids.shape[0], type_ids.device
        valid_one = torch.ones(batch, 1, dtype=torch.bool, device=device)
        status: Dict[str, str] = {}
        tokens: List[torch.Tensor] = [self._tag(image.global_embedding.embedding.unsqueeze(1), "global")]
        masks: List[torch.Tensor] = [valid_one]
        status["global"] = Status.AVAILABLE

        d_model = tokens[0].shape[-1]
        if image.roi.features is not None:
            tokens.append(self._tag(image.roi.features.unsqueeze(1), "roi")); masks.append(image.roi.valid.unsqueeze(1))
            status["roi"] = Status.AVAILABLE
        else:
            tokens.append(torch.zeros(batch, 1, d_model, device=device)); masks.append(~valid_one)
            status["roi"] = image.roi.status

        tokens.append(self._tag(self.type_embedding(type_ids).unsqueeze(1), "type")); masks.append(valid_one)
        status["type"] = Status.AVAILABLE

        if ontology.knowledge_embedding is not None and self.knowledge_projection is not None:
            tokens.append(self._tag(self.knowledge_projection(ontology.knowledge_embedding).unsqueeze(1), "knowledge"))
            masks.append(valid_one)
            status["knowledge"] = Status.AVAILABLE
        else:
            tokens.append(torch.zeros(batch, 1, d_model, device=device)); masks.append(~valid_one)
            status["knowledge"] = ontology.status
            if not self._logged_skip:
                log.warning("Ontology knowledge input skipped in fusion (status=%s); slot masked, nothing fabricated.", ontology.status)
                self._logged_skip = True

        patches = self._tag(image.vision.patch_features, "patch")
        tokens.append(patches); masks.append(torch.ones(batch, patches.shape[1], dtype=torch.bool, device=device))
        status["patch"] = Status.AVAILABLE
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1), status

    def forward(self, question: QuestionFeatures, image: ImageProcessingOutput, ontology: OntologyOutput,
                type_ids: torch.Tensor) -> FusionOutput:
        memory, memory_mask, status = self.build_memory(image, ontology, type_ids)
        queries = question.token_features
        for layer in self.layers:
            queries = layer(queries, memory, memory_mask)
        fused = self.output_norm(queries) * question.token_mask.unsqueeze(-1).to(queries.dtype)
        return FusionOutput(tokens=fused, mask=question.token_mask, modality_status=status)
