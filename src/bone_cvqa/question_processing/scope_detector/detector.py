"""Question Scope Detector — CHỈ CÓ Ở ONLINE (sơ đồ Offline không có). Baseline luật từ khóa, KHÔNG phải bộ phân loại đã huấn luyện.

Chính sách dự phòng khi ontology còn là placeholder:
  1. Có entity đã link -> in_scope.
  2. Khớp từ khóa xương/X-quang (nguyên từ; từ khóa kết thúc bằng '*' là khớp tiền tố) -> in_scope.
  3. Câu hỏi nối tiếp không có từ khóa nhưng các lượt trước khớp -> in_scope.
  4. Ngược lại -> `unmatched_policy` (mặc định out_of_scope), kèm lý do nêu rõ có thể từ chối nhầm.
"""
import re
from typing import Any, Dict, List, Optional

from ...commons.constants import Status
from ..language_encoder.conversation_text import normalize_history
from .keywords import DEFAULT_KEYWORDS

ONLINE_ONLY = True


class QuestionScopeDetector:
    def __init__(self, keywords: Optional[List[str]] = None, unmatched_policy: str = "out_of_scope") -> None:
        assert unmatched_policy in ("out_of_scope", "in_scope")
        self.keywords = [k.lower() for k in (keywords or DEFAULT_KEYWORDS)]
        self.unmatched_policy = unmatched_policy

    def _matches(self, text: str) -> List[str]:
        text = text.lower()
        found = []
        for keyword in self.keywords:
            body = re.escape(keyword[:-1]) + r"\w*" if keyword.endswith("*") else re.escape(keyword) + r"(?:e?s)?"
            if re.search(r"(?<!\w)" + body + r"(?!\w)", text):
                found.append(keyword)
        return found

    def detect(self, question: str, entity_result: Optional[Dict[str, Any]] = None,
               history: Optional[list] = None) -> Dict[str, Any]:
        if (entity_result or {}).get("entities"):
            return self._result("in_scope", "linked ontology entities found", "entity_link")
        matched = self._matches(question)
        if matched:
            return self._result("in_scope", f"matched domain keyword(s): {matched[:3]}", "keyword_fallback")
        earlier = " ".join(turn["content"] for turn in normalize_history(history))
        if earlier and self._matches(earlier):
            return self._result("in_scope", "follow-up question; earlier turns matched domain keywords", "keyword_history")
        reason = ("no bone/radiology keyword matched. Limitation: ontology entity linking is not implemented and the "
                  "keyword list is incomplete, so this may be a false rejection.")
        return self._result(self.unmatched_policy, reason, "keyword_fallback")

    @staticmethod
    def _result(label: str, reason: str, method: str) -> Dict[str, Any]:
        return {"label": label, "status": Status.AVAILABLE, "method": method, "reason": reason}
