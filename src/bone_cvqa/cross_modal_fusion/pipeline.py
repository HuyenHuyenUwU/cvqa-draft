"""Khối 4 — Cross-modal Fusion + Medical LLM Decoder (dùng chung Offline/Online)."""
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ..commons.types import (DecoderOutput, FusionOutput, ImageProcessingOutput, OntologyOutput, QuestionProcessingOutput)
from .cross_attention.fusion import CrossModalFusion
from .medical_llm_decoder.decoder import MedicalLLMDecoder
from .medical_llm_decoder.prompt_builder import build_llm_prompt


class CrossModalFusionPipeline(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        d_model, fusion_cfg, decoder_cfg = int(config["d_model"]), config["cross_modal_fusion"], config["decoder"]
        self.fusion = CrossModalFusion(d_model, fusion_cfg["n_layers"], fusion_cfg["n_heads"], fusion_cfg["dropout"])
        self.decoder = MedicalLLMDecoder(config["external_models"]["medical_llm"], d_model,
                                         decoder_cfg["max_prompt_tokens"], decoder_cfg["max_answer_tokens"])

    def fuse(self, question: QuestionProcessingOutput, image: ImageProcessingOutput, ontology: OntologyOutput) -> FusionOutput:
        type_ids = torch.tensor([p.type_id for p in question.type_prompts], device=question.features.pooled.device)
        return self.fusion(question.features, image, ontology, type_ids)

    def build_prompts(self, questions: List[str], histories: List[Any], question: QuestionProcessingOutput) -> List[str]:
        return [build_llm_prompt(q, h, tp.prompt_text) for q, h, tp in zip(questions, histories, question.type_prompts)]

    def teacher_forcing(self, fusion: FusionOutput, prompts: List[str], answers: List[Optional[str]]) -> DecoderOutput:
        return self.decoder.teacher_forcing(fusion, prompts, answers)

    def generate(self, fusion: FusionOutput, prompts: List[str], generation: Dict[str, Any]) -> List[str]:
        return self.decoder.generate(fusion, prompts, generation)
