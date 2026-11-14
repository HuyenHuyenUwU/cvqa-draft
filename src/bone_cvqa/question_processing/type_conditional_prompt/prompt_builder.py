"""Type-Conditional Prompt (khối 2): (câu hỏi, kết quả entity linking) -> TypePrompt. Độc lập với encoder."""
from typing import Any, Dict, Optional

from ...commons.constants import QuestionType
from ...commons.types import TypePrompt
from .question_type import classify_question_type

PROMPT_TEXT = {
    QuestionType.ANATOMY_LOCATION: "Task: state the anatomical location of the finding in the image.",
    QuestionType.FRACTURE_IDENTIFICATION: "Task: decide whether a fracture is visible and answer concisely.",
    QuestionType.DISEASE_IDENTIFICATION: "Task: identify the finding or disease category suggested by the image.",
    QuestionType.RADIOGRAPHIC_SIGN: "Task: describe the relevant radiographic sign visible in the image.",
    QuestionType.EXPLANATION: "Task: explain the answer by referring to what is visible in the image.",
    QuestionType.OTHER: "Task: answer the question about the image concisely.",
}


class TypeConditionalPromptBuilder:
    def build(self, question: str, entity_result: Optional[Dict[str, Any]] = None) -> TypePrompt:
        question_type, _ = classify_question_type(question)
        text = PROMPT_TEXT[question_type]
        entities = (entity_result or {}).get("entities", [])
        if entities:   # chỉ thêm khái niệm THẬT đã được link
            text += " Relevant concepts: " + ", ".join(str(e) for e in entities) + "."
        return TypePrompt(question_type=question_type, type_id=QuestionType.to_id(question_type), prompt_text=text)
