"""Hằng số dùng chung cho toàn bộ pipeline (trạng thái module, loại câu hỏi, chế độ chạy)."""
from enum import Enum


class Status:
    """Trạng thái trả về của mọi module có thể chưa hoàn thiện. Placeholder KHÔNG BAO GIỜ trả AVAILABLE."""
    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"   # placeholder: chưa viết thuật toán
    NOT_AVAILABLE = "not_available"       # đã viết nhưng thiếu checkpoint / dữ liệu
    SKIPPED = "skipped"                   # bỏ qua có chủ đích (vd. không có box để cắt ROI)
    FAILED = "failed"


class QuestionType:
    ANATOMY_LOCATION = "anatomy_location"
    FRACTURE_IDENTIFICATION = "fracture_identification"
    DISEASE_IDENTIFICATION = "disease_identification"
    RADIOGRAPHIC_SIGN = "radiographic_sign"
    EXPLANATION = "explanation"
    OTHER = "other"

    ALL = [ANATOMY_LOCATION, FRACTURE_IDENTIFICATION, DISEASE_IDENTIFICATION,
           RADIOGRAPHIC_SIGN, EXPLANATION, OTHER]

    @classmethod
    def to_id(cls, name: str) -> int:
        return cls.ALL.index(name) if name in cls.ALL else cls.ALL.index(cls.OTHER)


class ExecutionMode(str, Enum):
    """TRAIN = offline (bật gradient, train mode). INFER = online (tắt gradient, eval mode)."""
    TRAIN = "train"
    INFER = "infer"


ANSWER_SOURCE_EXPERT = "expert_annotation"
ANSWER_SOURCE_WEAK = "template_weak_supervision"   # câu hỏi-đáp sinh từ nhãn, KHÔNG phải ground truth chuyên gia
IGNORE_INDEX = -100
