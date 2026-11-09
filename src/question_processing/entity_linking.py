from __future__ import annotations

from ..common.logger import get_logger, log_placeholder
from ..common.schemas import Status

log = get_logger("question.entity_linking")


class PlaceholderQuestionEntityLinker:
    """[placeholder] Question-side Entity Linking (Question Processing block).
    in : question text (+ optional question embedding) ; out: {"entities": [], "status": "not_implemented"}"""

    def __init__(self):
        self._calls = 0

    def link(self, question_text: str, question_embedding=None):
        if self._calls == 0:
            log_placeholder(log, "Entity Linking (question side)")
        self._calls += 1
        return {"entities": [], "status": Status.NOT_IMPLEMENTED}
