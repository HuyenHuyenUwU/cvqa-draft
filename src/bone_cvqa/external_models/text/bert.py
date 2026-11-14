"""Question encoder: BERT (HuggingFace).  provider=huggingface (cần trọng số) | tiny_random (test offline).

predict({"texts": List[str]}) -> {
    "token_features": [B, T, H]   hidden states cuối,
    "token_mask":     [B, T] bool True = token thật,
    "pooled":         [B, H]      vector [CLS],
}
"""
from typing import Any, Dict

from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

from ..base import TorchModelAdapter
from ..clients.local_client import hf_source, load_hf_model
from ..registry import register_adapter
from .hash_tokenizer import TinyHashTokenizer


@register_adapter("question_encoder", "huggingface", "tiny_random")
class BertTextAdapter(TorchModelAdapter):
    DEFAULT_NAME = "bert-base-uncased"

    def load(self) -> None:
        self.max_length = int(self.config.get("max_length", 128))
        if self.config.get("provider") == "tiny_random":
            hidden = int(self.config.get("hidden_size", 48))
            bert_config = BertConfig(vocab_size=512, hidden_size=hidden, num_hidden_layers=2, num_attention_heads=2,
                                     intermediate_size=64, max_position_embeddings=256, pad_token_id=0)
            self.model = BertModel(bert_config)
            self.tokenizer = TinyHashTokenizer(512)
        else:
            self.config.setdefault("name", self.DEFAULT_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(**hf_source(self.config))
            self.model = load_hf_model(AutoModel, self.config)
        self.hidden_size = self.model.config.hidden_size
        self._finish_load()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        encoded = self.tokenizer(inputs["texts"], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        hidden = self.model(**encoded).last_hidden_state
        return {"token_features": hidden, "token_mask": encoded["attention_mask"].bool(), "pooled": hidden[:, 0]}
