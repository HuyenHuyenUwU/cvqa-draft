"""[PLACEHOLDER] Entity Linking phía câu hỏi.  in: câu hỏi  ->  out: {"entities": [], "status": "not_implemented"}"""
from typing import Any, Dict

from ...commons.constants import Status
from ...commons.placeholder import PlaceholderModule


class QuestionEntityLinker(PlaceholderModule):
    module_name = "Entity Linking (question side)"

    def empty_result(self) -> Dict[str, Any]:
        return {"entities": [], "status": Status.NOT_IMPLEMENTED}

    def link(self, question: str) -> Dict[str, Any]:
        return self.run(question)
