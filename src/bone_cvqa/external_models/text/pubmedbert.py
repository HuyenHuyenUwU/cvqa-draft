"""Ablation: PubMedBERT/BiomedBERT làm question encoder (cùng giao diện với BertTextAdapter)."""
from ..registry import register_adapter
from .bert import BertTextAdapter


@register_adapter("question_encoder", "pubmedbert")
class PubMedBertTextAdapter(BertTextAdapter):
    DEFAULT_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
