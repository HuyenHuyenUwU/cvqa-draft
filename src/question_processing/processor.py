"""Shared Question Processing glue (used identically by Offline and Online):
   conversation text -> question-side Entity Linking (placeholder) -> Type-Conditional Prompt.
The (neural) Conversation-aware Language Encoder is run inside CVQAModel."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..data.text_processing import build_conversation_text
from .entity_linking import PlaceholderQuestionEntityLinker
from .type_conditional_prompt import TypeConditionalPromptBuilder


class QuestionProcessor:
    def __init__(self, entity_linker=None, prompt_builder=None):
        self.entity_linker = entity_linker or PlaceholderQuestionEntityLinker()
        self.prompt_builder = prompt_builder or TypeConditionalPromptBuilder()

    def process(self, question: str, history=None) -> Dict[str, Any]:
        entities = self.entity_linker.link(question)
        prompt = self.prompt_builder.build(question, entities)
        return {"conversation_text": build_conversation_text(question, history), "entities": entities,
                "type_prompt": prompt}
