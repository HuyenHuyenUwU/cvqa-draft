"""Medical LLM decoder: LLM mã nguồn mở chạy local (HuggingFace CausalLM).
provider = huggingface | local | tiny_random (Qwen2 nhỏ, trọng số ngẫu nhiên, chỉ để test).

predict({"mode": ...}) hỗ trợ:
  "embed"    {"input_ids": List[int]}                       -> {"embeddings": [L, H]}
  "hidden"   {"inputs_embeds": [B,L,H], "attention_mask"}   -> {"hidden": [B,L,H]}
  "lm_head"  {"hidden": [M,H]}                               -> {"logits": [M,V]}     (Answer Head)
  "generate" {"inputs_embeds", "attention_mask", **gen}      -> {"texts": List[str]}
"""
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from ..base import TorchModelAdapter
from ..clients.local_client import hf_source, load_hf_model
from ..registry import register_adapter
from ..text.hash_tokenizer import TinyHashTokenizer


@register_adapter("medical_llm", "huggingface", "local", "tiny_random")
class HFCausalLMAdapter(TorchModelAdapter):
    def load(self) -> None:
        if self.config.get("provider") == "tiny_random":
            llm_config = Qwen2Config(vocab_size=512, hidden_size=int(self.config.get("hidden_size", 64)),
                                     intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, max_position_embeddings=2048, eos_token_id=2,
                                     pad_token_id=0, tie_word_embeddings=True)
            self.model = Qwen2ForCausalLM(llm_config)
            self.tokenizer = TinyHashTokenizer(512)
        else:
            dtype = getattr(torch, self.config.get("dtype", "float32"))
            self.tokenizer = AutoTokenizer.from_pretrained(**hf_source(self.config))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = load_hf_model(AutoModelForCausalLM, self.config, dtype=dtype)
        self.hidden_size = self.model.config.hidden_size
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token_id
        self._finish_load()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mode = inputs["mode"]
        if mode == "embed":
            ids = torch.tensor(inputs["input_ids"], dtype=torch.long, device=self.device)
            return {"embeddings": self.model.get_input_embeddings()(ids)}
        if mode == "hidden":
            out = self.model.base_model(inputs_embeds=inputs["inputs_embeds"], attention_mask=inputs["attention_mask"])
            return {"hidden": out.last_hidden_state}
        if mode == "lm_head":
            return {"logits": self.model.get_output_embeddings()(inputs["hidden"])}
        if mode == "generate":
            out = self.model.generate(
                inputs_embeds=inputs["inputs_embeds"], attention_mask=inputs["attention_mask"],
                max_new_tokens=inputs.get("max_new_tokens", 32), do_sample=inputs.get("do_sample", False),
                temperature=inputs.get("temperature") if inputs.get("do_sample") else None,
                top_p=inputs.get("top_p") if inputs.get("do_sample") else None,
                pad_token_id=self.pad_token_id, eos_token_id=self.eos_token_id)
            return {"texts": [self.tokenizer.decode(s, skip_special_tokens=True).strip() for s in out]}
        raise ValueError(f"Unknown mode '{mode}'")
