"""Hợp đồng dữ liệu (input/output) giữa các module. Mọi tensor có batch ở chiều 0."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class PreprocessedImage:
    tensor: torch.Tensor            # [3, S, S] đã chuẩn hóa, đưa vào vision encoder
    gray_unit: np.ndarray           # [S, S] float32 trong [0,1], min-max (không kéo giãn tương phản) - dùng cho image quality
    original_size_hw: Tuple[int, int]


@dataclass
class VisionFeatures:
    patch_features: torch.Tensor    # [B, N, d_model] (đã project)
    global_feature: torch.Tensor    # [B, D_backbone] (chiều gốc của backbone, chưa project)
    grid_size: Tuple[int, int]      # (h, w), h*w = N


@dataclass
class LocalizationResult:
    status: str
    box_xyxy_norm: Optional[torch.Tensor] = None   # [B, 4] chuẩn hóa theo ảnh, hoặc None
    heatmap: Optional[torch.Tensor] = None         # [B, N] hoặc None


@dataclass
class RoiFeatures:
    status: str
    features: Optional[torch.Tensor] = None        # [B, D] hoặc None
    valid: Optional[torch.Tensor] = None           # [B] bool: mẫu nào có ROI thật


@dataclass
class GlobalEmbedding:
    embedding: torch.Tensor                         # [B, D]


@dataclass
class ImageProcessingOutput:
    vision: VisionFeatures
    localization: LocalizationResult
    roi: RoiFeatures
    global_embedding: GlobalEmbedding


@dataclass
class QuestionFeatures:
    token_features: torch.Tensor    # [B, T, D]
    token_mask: torch.Tensor        # [B, T] bool, True = token thật
    pooled: torch.Tensor            # [B, D]


@dataclass
class TypePrompt:
    question_type: str
    type_id: int
    prompt_text: str
    method: str = "rule_based_keywords"   # KHÔNG phải mô hình học được


@dataclass
class QuestionProcessingOutput:
    features: QuestionFeatures
    conversation_texts: List[str]
    entities: List[Dict[str, Any]]              # mỗi mẫu: {"entities": [], "status": ...}
    type_prompts: List[TypePrompt]
    scope: Optional[List[Dict[str, Any]]] = None   # chỉ có khi chạy ONLINE (Question Scope Detector)


@dataclass
class OntologyOutput:
    status: str
    subgraph: Dict[str, Any] = field(default_factory=dict)
    linked_entities: Dict[str, Any] = field(default_factory=dict)
    knowledge_embedding: Optional[torch.Tensor] = None   # None = chưa có (fusion sẽ che mask)


@dataclass
class FusionOutput:
    tokens: torch.Tensor            # [B, T, d_model]
    mask: torch.Tensor              # [B, T] bool
    modality_status: Dict[str, str]


@dataclass
class DecoderOutput:
    hidden: torch.Tensor            # [B, L, H_llm]
    labels: torch.Tensor            # [B, L], IGNORE_INDEX ngoài vị trí đáp án
    attention_mask: torch.Tensor    # [B, L]


@dataclass
class SystemContext:
    """Toàn bộ kết quả trung gian của khối 1-4 (dùng chung Offline/Online)."""
    question: QuestionProcessingOutput
    image: Optional[ImageProcessingOutput] = None
    ontology: Optional[OntologyOutput] = None
    fusion: Optional[FusionOutput] = None
    prompts: Optional[List[str]] = None       # prompt văn bản đưa vào LLM
    early_exit_reason: Optional[str] = None   # vd. "out_of_scope"
