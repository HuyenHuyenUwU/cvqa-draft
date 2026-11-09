"""Medical LLM decoder body + embedding-level multimodal interface.

Sequence fed to the LLM (through ``inputs_embeds``):
    [ soft multimodal tokens (n) ][ prompt token embeddings ][ answer token embeddings + EOS ]
labels are -100 everywhere except the answer tokens. The LLM is frozen by default (see ``freeze_llm``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ...data.text_processing import build_conversation_text

DEFAULT_SYSTEM = ("You are an assistant that answers questions about bone X-ray images. "
                  "Answer concisely and only from the image evidence provided.")


class LLMDecoder(nn.Module):
    def __init__(self, llm: nn.Module, tokenizer: Any, max_prompt_tokens: int = 256,
                 max_answer_tokens: int = 64, system_prompt: str = DEFAULT_SYSTEM):
        super().__init__()
        self.llm, self.tok = llm, tokenizer
        self.max_prompt, self.max_answer, self.system = max_prompt_tokens, max_answer_tokens, system_prompt
        self.hidden_size = llm.config.hidden_size
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (self.eos or 0)

    # ---- text side ----
    def build_prompt(self, question: str, history, type_prompt: str) -> str:
        body = build_conversation_text(question, history)
        return f"{self.system}\n{type_prompt}\n{body}\nAnswer:"

    def _ids(self, text: str, max_len: int, keep_tail: bool) -> List[int]:
        ids = list(self.tok(text, add_special_tokens=False)["input_ids"])
        if len(ids) > max_len:
            ids = ids[-max_len:] if keep_tail else ids[:max_len]
        return ids

    def _embed(self, ids: List[int], device) -> torch.Tensor:
        return self.llm.get_input_embeddings()(torch.tensor(ids, dtype=torch.long, device=device))

    # ---- sequence assembly ----
    def _assemble(self, soft: torch.Tensor, prompts: List[str], answers: Optional[List[Optional[str]]], left_pad: bool):
        dev, emb_dtype = soft.device, self.llm.get_input_embeddings().weight.dtype
        soft = soft.to(emb_dtype)
        seqs, labs = [], []
        for i, p in enumerate(prompts):
            pe = self._embed(self._ids(p, self.max_prompt, keep_tail=True), dev)
            parts, lab = [soft[i], pe], [-100] * (soft.shape[1] + pe.shape[0])
            if answers is not None and answers[i] is not None:
                a_ids = self._ids(answers[i], self.max_answer - 1, keep_tail=False) + [self.eos]
                parts.append(self._embed(a_ids, dev))
                lab += a_ids
            seqs.append(torch.cat(parts, 0))
            labs.append(lab)
        L, H = max(s.shape[0] for s in seqs), soft.shape[-1]
        emb = torch.zeros(len(seqs), L, H, dtype=emb_dtype, device=dev)
        att = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
        lbl = torch.full((len(seqs), L), -100, dtype=torch.long, device=dev)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            n = s.shape[0]
            sl = slice(L - n, L) if left_pad else slice(0, n)
            emb[i, sl], att[i, sl], lbl[i, sl] = s, 1, torch.tensor(l, device=dev)
        return emb, att, lbl

    def forward(self, soft: torch.Tensor, prompts: List[str], answers: List[Optional[str]]) -> Dict[str, torch.Tensor]:
        """Training / teacher-forced pass -> hidden states of the decoder body (logits come from AnswerHead)."""
        emb, att, lbl = self._assemble(soft, prompts, answers, left_pad=False)
        hidden = self.llm.base_model(inputs_embeds=emb, attention_mask=att).last_hidden_state
        return {"hidden": hidden, "labels": lbl, "attention_mask": att}

    @torch.no_grad()
    def generate(self, soft: torch.Tensor, prompts: List[str], max_new_tokens: Optional[int] = None,
                 do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0,
                 return_scores: bool = False) -> Dict[str, Any]:
        emb, att, _ = self._assemble(soft, prompts, None, left_pad=True)
        out = self.llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=max_new_tokens or self.max_answer,
                                do_sample=do_sample, temperature=temperature if do_sample else None,
                                top_p=top_p if do_sample else None, pad_token_id=self.pad, eos_token_id=self.eos,
                                output_scores=return_scores, return_dict_in_generate=True)
        seqs = out.sequences
        texts = [self.tok.decode(s, skip_special_tokens=True).strip() for s in seqs]
        return {"texts": texts, "scores": getattr(out, "scores", None) if return_scores else None}
