"""Prompt văn bản đưa vào LLM (kèm Type-Conditional Prompt). Đáp án KHÔNG nằm trong prompt."""
from typing import Any, Optional

from ...question_processing.language_encoder.conversation_text import build_conversation_text

SYSTEM_PROMPT = ("You are an assistant that answers questions about bone X-ray images. "
                 "Answer concisely and only from the image evidence provided.")


def build_llm_prompt(question: str, history: Optional[Any], type_prompt_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n{type_prompt_text}\n{build_conversation_text(question, history)}\nAnswer:"
