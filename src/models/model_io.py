"""Model construction + inference-grade checkpoint I/O, shared by Offline and Online.
(Online imports THIS module, never the Offline trainer.)

Only parameters with requires_grad=True are saved (fusion, adapter, localizer, any unfrozen encoder part);
frozen pretrained towers/LLM are rebuilt from their own local checkpoints, which keeps checkpoints small.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ..common.exceptions import CheckpointError
from ..common.logger import get_logger
from .cvqa_model import CVQAModel
from .language.text_encoder import ConversationTextEncoder
from .llm.llm_decoder import LLMDecoder
from .llm.llm_loader import freeze_llm, load_llm
from .visual.biomedclip_encoder import BiomedCLIPVisualEncoder, load_biomedclip

log = get_logger("models.io")
FORMAT_VERSION = 1
MODEL_KEYS = ["model_source", "preprocess", "biomedclip", "text_encoder", "llm", "fusion", "localization", "generation"]


def model_config_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: cfg[k] for k in MODEL_KEYS if k in cfg}


def build_cvqa_model(cfg: Dict[str, Any]) -> CVQAModel:
    src = cfg.get("model_source", "pretrained")
    size = int(cfg["preprocess"]["image_size"])
    bc, tc, lc = cfg["biomedclip"], cfg.get("text_encoder", {}), cfg["llm"]
    if src == "tiny_random":                                   # tests / smoke test only
        from .tiny_factory import build_tiny_components
        clip, clip_tok, llm, llm_tok = build_tiny_components(image_size=size)
        ctx = clip_tok.context_length
    elif src == "pretrained":
        clip, clip_tok = load_biomedclip(bc)
        llm, llm_tok = load_llm(lc)
        ctx = int(bc.get("context_length", 256))
    else:
        raise CheckpointError(f"Unknown model_source '{src}'")
    visual = BiomedCLIPVisualEncoder(clip.visual, size, bool(bc.get("freeze", True)),
                                     int(bc.get("unfreeze_last_n_blocks", 0)))
    text = ConversationTextEncoder("biomedclip", clip.text, clip_tok, min(int(tc.get("max_length", ctx)), ctx),
                                   pad_token_id=0, freeze=bool(tc.get("freeze", True)))
    if lc.get("freeze", True):
        freeze_llm(llm)
    decoder = LLMDecoder(llm, llm_tok, int(lc.get("max_prompt_tokens", 256)), int(lc.get("max_answer_tokens", 64)))
    return CVQAModel(visual, text, decoder, cfg.get("fusion", {}), cfg.get("localization", {}))


def trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def save_model_checkpoint(path, model: CVQAModel, cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, "model_state": trainable_state_dict(model),
               "model_config": model_config_subset(cfg), "localizer_trained": bool(model.localizer_trained)}
    payload.update(extra or {})
    torch.save(payload, path)


def load_model_checkpoint(path, cfg_override: Optional[Dict[str, Any]] = None,
                          map_location="cpu") -> Tuple[CVQAModel, Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"Checkpoint not found: {p}")
    ck = torch.load(p, map_location=map_location, weights_only=False)
    if ck.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"Unsupported checkpoint format {ck.get('format_version')}")
    cfg = dict(ck["model_config"])
    for k, v in (cfg_override or {}).items():       # e.g. local_dir / device paths valid on THIS machine
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    model = build_cvqa_model(cfg)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    trainable_missing = [k for k in missing if k in {n for n, q in model.named_parameters() if q.requires_grad}]
    if unexpected or trainable_missing:
        raise CheckpointError(f"State mismatch. unexpected={unexpected[:5]} missing_trainable={trainable_missing[:5]}")
    model.localizer_trained = bool(ck.get("localizer_trained", False))
    return model.eval(), ck
