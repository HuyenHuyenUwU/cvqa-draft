"""Ablation: MedGemma-4b-it (google/medgemma-4b-it, model gated trên HuggingFace). Dùng như LLM chỉ-văn-bản.
Lưu ý: MedGemma là checkpoint Gemma3 đa phương thức; nếu AutoModelForCausalLM không nạp được thì cần
adapter riêng. Chưa kiểm thử với trọng số thật."""
from ..registry import register_adapter
from .hf_causal_lm import HFCausalLMAdapter


@register_adapter("medical_llm", "medgemma")
class MedGemmaAdapter(HFCausalLMAdapter):
    def load(self) -> None:
        self.config.setdefault("name", "google/medgemma-4b-it")
        super().load()
