"""Phân loại loại câu hỏi bằng LUẬT TỪ KHÓA (heuristic minh bạch, KHÔNG phải mô hình hiểu ý định đã huấn luyện).
Luật kiểm tra theo thứ tự; khớp đầu tiên thắng."""
import re
from typing import List, Tuple

from ...commons.constants import QuestionType

RULES: List[Tuple[str, List[str]]] = [
    (QuestionType.EXPLANATION, ["why", "explain", "reason", "because", "how do you know", "giải thích", "tại sao"]),
    (QuestionType.ANATOMY_LOCATION, ["where", "which bone", "which part", "which region", "location", "located",
                                     "anatomical", "ở đâu", "vị trí", "xương nào"]),
    (QuestionType.FRACTURE_IDENTIFICATION, ["fracture", "broken", "break", "crack", "gãy"]),
    (QuestionType.RADIOGRAPHIC_SIGN, ["sign", "appearance", "density", "lucen", "sclerosis", "periosteal", "cortex",
                                      "cortical", "margin", "pattern", "dấu hiệu"]),
    (QuestionType.DISEASE_IDENTIFICATION, ["disease", "tumor", "tumour", "lesion", "malignan", "benign", "diagnos",
                                           "osteo", "cancer", "condition", "category", "finding", "bệnh", "u xương"]),
]


def classify_question_type(question: str) -> Tuple[str, List[str]]:
    text = question.lower()
    for question_type, keywords in RULES:
        matched = [k for k in keywords if re.search(r"\b" + re.escape(k), text)]
        if matched:
            return question_type, matched
    return QuestionType.OTHER, []
