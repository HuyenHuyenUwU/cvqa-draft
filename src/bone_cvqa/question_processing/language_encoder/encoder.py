"""Conversation-aware Language Encoder (khối 2): văn bản hội thoại -> QuestionFeatures (đã project về d_model)."""
from typing import Any, Dict, List

from torch import nn

from ...commons.types import QuestionFeatures
from ...external_models import build_adapter
from ..language_encoder.conversation_text import build_conversation_text
from ...image_processing.vision_encoder.projection import FeatureProjection


class ConversationAwareLanguageEncoder(nn.Module):
    def __init__(self, external_config: Dict[str, Any], d_model: int) -> None:
        super().__init__()
        self.backbone = build_adapter("question_encoder", external_config)
        self.token_projection = FeatureProjection(self.backbone.hidden_size, d_model)
        self.pooled_projection = FeatureProjection(self.backbone.hidden_size, d_model)

    def build_texts(self, questions: List[str], histories: List[Any]) -> List[str]:
        return [build_conversation_text(q, h) for q, h in zip(questions, histories)]

    def forward(self, conversation_texts: List[str]) -> QuestionFeatures:
        raw = self.backbone.predict({"texts": conversation_texts})
        return QuestionFeatures(token_features=self.token_projection(raw["token_features"]),
                                token_mask=raw["token_mask"], pooled=self.pooled_projection(raw["pooled"]))
