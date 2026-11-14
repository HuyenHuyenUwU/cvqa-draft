"""BoneCVQASystem: lắp 5 khối dùng chung Offline và Online (Image, Question, Ontology, Fusion+Decoder, Answer).

Chỉ chứa phần tính toán thuần (forward). KHÔNG có loss/optimizer (nằm ở bone_cvqa.training) và KHÔNG có
quality/scope/uncertainty/warning (nằm ở bone_cvqa.inference). Gradient do `commons.execution_mode` quyết định:
  Offline: with execution_context(system, TRAIN)  -> train mode + bật gradient
  Online : with execution_context(system, INFER)  -> eval mode + tắt gradient
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .answer_generation.pipeline import AnswerGenerationPipeline
from .commons.device import resolve_device
from .commons.types import SystemContext
from .cross_modal_fusion.pipeline import CrossModalFusionPipeline
from .image_processing.pipeline import ImageProcessingPipeline
from .ontology_processing.pipeline import OntologyProcessingPipeline
from .question_processing.pipeline import QuestionProcessingPipeline


class BoneCVQASystem(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.image_processing = ImageProcessingPipeline(config)
        self.question_processing = QuestionProcessingPipeline(config)
        self.cross_modal_fusion = CrossModalFusionPipeline(config)
        self.ontology_processing = OntologyProcessingPipeline(config)      # placeholder, không có tham số
        self.answer_generation = AnswerGenerationPipeline(self.cross_modal_fusion)

    def encode(self, images: torch.Tensor, questions: List[str], histories: List[Any],
               scope_detector: Optional[Any] = None) -> SystemContext:
        """Khối 1-4 (tới hết Cross-modal Fusion). `scope_detector` chỉ truyền ở Online; nếu mọi mẫu out_of_scope thì dừng sớm."""
        question = self.question_processing(questions, histories, scope_detector)
        if question.scope and all(item["label"] == "out_of_scope" for item in question.scope):
            return SystemContext(question=question, early_exit_reason="out_of_scope")
        image = self.image_processing(images)
        ontology = self.ontology_processing(image, question.entities)
        fusion = self.cross_modal_fusion.fuse(question, image, ontology)
        prompts = self.cross_modal_fusion.build_prompts(questions, histories, question)
        return SystemContext(question=question, image=image, ontology=ontology, fusion=fusion, prompts=prompts)

    def answer_logits(self, context: SystemContext, answers: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher forcing: -> (logits [M,V], targets [M]) cho loss đáp án (dùng ở Offline)."""
        decoder_output = self.cross_modal_fusion.teacher_forcing(context.fusion, context.prompts, answers)
        return self.answer_generation.answer_logits(decoder_output)

    def generate_answers(self, context: SystemContext, generation: Dict[str, Any]) -> List[str]:
        return self.answer_generation.generate_answers(context.fusion, context.prompts, generation)


def build_system(config: Dict[str, Any]) -> BoneCVQASystem:
    return BoneCVQASystem(config).to(resolve_device(config.get("device", "auto")))
