"""CVQAModel: the neural graph shared by Offline and Online (no losses, no optimiser here).

  images -> VisualEncoder -> {patch feats, global emb}
        patch feats -> LesionLocalizer -> box ; box + patch feats -> RoI features
  texts  -> ConversationTextEncoder -> {token feats, pooled emb}
  {global, RoI, question, type id, knowledge(optional)} -> MultimodalFusion -> tokens
  tokens -> LLMAdapter -> soft tokens -> LLMDecoder (body) -> AnswerHead (logits)  | generate()

Question Processing glue (entity linking / type prompt) and the ontology placeholders are orchestrated by
the pipelines, which pass their results in. Nothing about training lives here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .fusion.llm_adapter import LLMAdapter
from .fusion.multimodal_fusion import FusionInputs, MultimodalFusion
from .language.text_encoder import ConversationTextEncoder
from .llm.answer_head import AnswerHead
from .llm.llm_decoder import LLMDecoder
from .localization.lesion_localizer import LesionLocalizer, RoIFeatureExtractor
from .visual.biomedclip_encoder import BiomedCLIPVisualEncoder


class CVQAModel(nn.Module):
    def __init__(self, visual: BiomedCLIPVisualEncoder, text: ConversationTextEncoder, decoder: LLMDecoder,
                 fusion_cfg: Dict[str, Any], localizer_cfg: Optional[Dict[str, Any]] = None,
                 num_question_types: int = 6):
        super().__init__()
        self.visual, self.text, self.decoder = visual, text, decoder
        has_patches = visual.patch_dim is not None
        dims = {"global": visual.global_dim, "roi": visual.patch_dim or visual.global_dim,
                "question": text.embedding_dim, "question_tokens": text.feature_dim, "patches": visual.patch_dim,
                "knowledge": fusion_cfg.get("knowledge_dim")}
        d = int(fusion_cfg.get("d_model", 256))
        self.fusion = MultimodalFusion(dims, d, int(fusion_cfg.get("n_layers", 2)), int(fusion_cfg.get("n_heads", 4)),
                                       float(fusion_cfg.get("dropout", 0.1)), num_question_types,
                                       bool(fusion_cfg.get("use_patch_tokens", False)) and has_patches)
        self.adapter = LLMAdapter(d, decoder.hidden_size, int(fusion_cfg.get("n_soft_tokens", 8)),
                                  int(fusion_cfg.get("adapter_heads", 4)))
        self.answer_head = AnswerHead(decoder.llm.get_output_embeddings())
        use_loc = bool((localizer_cfg or {}).get("enabled", True)) and has_patches
        self.localizer = LesionLocalizer(visual.patch_dim, int((localizer_cfg or {}).get("hidden", 256))) if use_loc else None
        self.roi = RoIFeatureExtractor()
        self.localizer_trained = False        # set True by the trainer once box supervision has been used

    # ------------------------------------------------------------------
    def _multimodal(self, images, conv_texts, type_ids, gt_box=None, has_box=None, use_gt_roi=False,
                    knowledge: Optional[Dict[str, Any]] = None):
        vis = self.visual(images)
        B, dev = images.shape[0], images.device
        loc, box, valid = None, torch.zeros(B, 4, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
        if self.localizer is not None and vis.patch_features is not None:
            loc = self.localizer(vis.patch_features, vis.grid_size)
            if use_gt_roi and gt_box is not None:
                box, valid = gt_box.to(dev).clone(), has_box.to(dev).clone()
            if self.localizer_trained:                                   # predicted boxes only if trained
                rows = ~valid
                box = torch.where(rows[:, None], loc["box_xyxy_norm"].detach(), box)
                valid = valid | rows
        roi_feat = None
        if vis.patch_features is not None:
            roi_feat, valid = self.roi(vis.patch_features, vis.grid_size, box, valid)
        txt = self.text(conv_texts)
        kn = knowledge or {}
        f = FusionInputs(global_embedding=vis.global_embedding, question_embedding=txt.question_embedding,
                         question_tokens=txt.question_features, question_mask=txt.question_mask,
                         type_id=type_ids.to(dev), roi_features=roi_feat, roi_valid=valid if roi_feat is not None else None,
                         knowledge_embedding=kn.get("embedding"), knowledge_status=kn.get("status", "not_implemented"),
                         patch_features=vis.patch_features)
        fo = self.fusion(f)
        soft = self.adapter(fo.tokens, fo.mask)
        return {"soft": soft, "fusion": fo, "loc": loc, "roi_valid": valid, "visual": vis}

    def forward(self, images, conv_texts: List[str], type_ids, llm_prompts: List[str],
                answers: Optional[List[Optional[str]]] = None, gt_box=None, has_box=None, use_gt_roi=False,
                knowledge=None) -> Dict[str, Any]:
        mm = self._multimodal(images, conv_texts, type_ids, gt_box, has_box, use_gt_roi, knowledge)
        out = {"loc": mm["loc"], "roi_valid": mm["roi_valid"], "fusion_status": mm["fusion"].modality_status,
               "grid_size": mm["visual"].grid_size}
        if answers is not None:
            dec = self.decoder(mm["soft"], llm_prompts, answers)
            logits, targets = self.answer_head(dec["hidden"], dec["labels"])
            out.update({"answer_logits": logits, "answer_targets": targets})
        return out

    @torch.no_grad()
    def generate(self, images, conv_texts, type_ids, llm_prompts, knowledge=None, **gen_kw) -> Dict[str, Any]:
        mm = self._multimodal(images, conv_texts, type_ids, knowledge=knowledge)
        g = self.decoder.generate(mm["soft"], llm_prompts, **gen_kw)
        return {"texts": g["texts"], "scores": g["scores"], "loc": mm["loc"], "roi_valid": mm["roi_valid"],
                "fusion_status": mm["fusion"].modality_status, "grid_size": mm["visual"].grid_size,
                "localizer_trained": self.localizer_trained}
