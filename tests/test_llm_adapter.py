import torch

from src.models.fusion.llm_adapter import LLMAdapter
from src.models.tiny_factory import DummyLLMTokenizer, build_tiny_llm
from src.models.llm.llm_decoder import LLMDecoder
from src.models.llm.answer_head import AnswerHead


def test_adapter_output_matches_llm_hidden():
    llm = build_tiny_llm(hidden=64)
    ad = LLMAdapter(32, llm.config.hidden_size, n_soft_tokens=4, n_heads=2)
    tokens, mask = torch.randn(2, 9, 32), torch.ones(2, 9, dtype=torch.bool)
    mask[1, 5:] = False
    s = ad(tokens, mask)
    assert s.shape == (2, 4, 64)


def test_decoder_forward_head_and_generate():
    llm, tok = build_tiny_llm(), DummyLLMTokenizer()
    dec = LLMDecoder(llm, tok, max_prompt_tokens=32, max_answer_tokens=8)
    head = AnswerHead(llm.get_output_embeddings())
    soft = torch.randn(2, 4, 64, requires_grad=True)
    prompts = [dec.build_prompt("is there a fracture ?", [], "Task: x"), dec.build_prompt("where ?", [("q", "a")], "Task: y")]
    out = dec(soft, prompts, ["yes a fracture is present", None])
    logits, targets = head(out["hidden"], out["labels"])
    assert logits.shape[0] == targets.shape[0] > 0 and logits.shape[1] == llm.config.vocab_size
    # row 2 has no answer -> contributes no supervised positions
    assert (out["labels"][1] == -100).all()
    # gradients flow through the frozen LLM into the soft tokens
    for p in llm.parameters():
        p.requires_grad = False
    out = dec(soft, prompts, ["yes", "no"])
    lg, tg = head(out["hidden"], out["labels"])
    torch.nn.functional.cross_entropy(lg, tg).backward()
    assert soft.grad is not None and soft.grad.abs().sum() > 0
    g = dec.generate(soft.detach(), prompts, max_new_tokens=5)
    assert len(g["texts"]) == 2 and all(isinstance(t, str) for t in g["texts"])
