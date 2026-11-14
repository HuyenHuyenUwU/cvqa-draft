"""Khối 2 — Question Processing (dùng chung Offline/Online).

    Conversation-aware Language Encoder -> Entity Linking -> [Question Scope Detector: CHỈ Online] -> Type-Conditional Prompt

`scope_detector=None` (Offline) bỏ qua bước scope; Online truyền detector vào. Detector được tiêm từ ngoài nên module
chỉ-Online không bị import trong đường chạy Offline.
"""
from typing import Any, List, Optional

from torch import nn

from ..commons.types import QuestionProcessingOutput
from .entity_linking.linker import QuestionEntityLinker
from .language_encoder.encoder import ConversationAwareLanguageEncoder
from .type_conditional_prompt.prompt_builder import TypeConditionalPromptBuilder


class QuestionProcessingPipeline(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.language_encoder = ConversationAwareLanguageEncoder(config["external_models"]["question_encoder"],
                                                                 int(config["d_model"]))
        self.entity_linker = QuestionEntityLinker()
        self.prompt_builder = TypeConditionalPromptBuilder()

    def forward(self, questions: List[str], histories: List[Any], scope_detector: Optional[Any] = None) -> QuestionProcessingOutput:
        texts = self.language_encoder.build_texts(questions, histories)
        features = self.language_encoder(texts)
        entities = [self.entity_linker.link(q) for q in questions]
        scope = None
        if scope_detector is not None:
            scope = [scope_detector.detect(q, e, h) for q, e, h in zip(questions, entities, histories)]
        prompts = [self.prompt_builder.build(q, e) for q, e in zip(questions, entities)]
        return QuestionProcessingOutput(features=features, conversation_texts=texts, entities=entities,
                                        type_prompts=prompts, scope=scope)
