"""Khối 5 — Answer Generation (dùng chung Offline/Online).
Answer (chạy thật) + Heatmap/BBox, Evidence Graph, Visual+Knowledge Explanation (placeholder)."""
from typing import Any, Dict, List, Tuple

import torch

from ..commons.types import DecoderOutput, FusionOutput
from .answer.answer_head import AnswerHead
from .answer.generator import AnswerGenerator
from .evidence_graph.predictor import EvidenceGraphPredictor
from .heatmap_bbox.predictor import HeatmapBoundingBoxPredictor
from .visual_knowledge_explanation.explainer import VisualKnowledgeExplainer


class AnswerGenerationPipeline:
    def __init__(self, fusion_pipeline) -> None:
        self.answer_head = AnswerHead(fusion_pipeline.decoder.llm)
        self.generator = AnswerGenerator(fusion_pipeline.decoder)
        self.heatmap_bbox = HeatmapBoundingBoxPredictor()
        self.evidence_graph = EvidenceGraphPredictor()
        self.explainer = VisualKnowledgeExplainer()

    def answer_logits(self, decoder_output: DecoderOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.answer_head.logits_at_answer_positions(decoder_output)

    def generate_answers(self, fusion: FusionOutput, prompts: List[str], generation: Dict[str, Any]) -> List[str]:
        return self.generator.generate(fusion, prompts, generation)

    def placeholder_outputs(self) -> Dict[str, Dict[str, Any]]:
        return {"localization": self.heatmap_bbox.run(), "evidence_graph": self.evidence_graph.run(),
                "visual_knowledge_explanation": self.explainer.run()}
