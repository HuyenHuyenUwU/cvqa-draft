import torch

from src.models.fusion.multimodal_fusion import FusionInputs, MultimodalFusion
from src.models.localization.lesion_localizer import LesionLocalizer, RoIFeatureExtractor, box_to_patch_mask


def _inputs(B=2, T=7, with_roi=True):
    return FusionInputs(global_embedding=torch.randn(B, 20), question_embedding=torch.randn(B, 12),
                        question_tokens=torch.randn(B, T, 16), question_mask=torch.ones(B, T, dtype=torch.bool),
                        type_id=torch.tensor([0, 3][:B]),
                        roi_features=torch.randn(B, 24) if with_roi else None,
                        roi_valid=torch.tensor([True, False][:B]) if with_roi else None)


def _fusion(**kw):
    return MultimodalFusion({"global": 20, "roi": 24, "question": 12, "question_tokens": 16, "patches": 24,
                             "knowledge": None}, d_model=32, n_layers=1, n_heads=2, **kw)


def test_fusion_dimensions_and_masking():
    f = _fusion()
    o = f(_inputs())
    assert o.tokens.shape == (2, 5 + 7, 32) and o.mask.shape == (2, 12)
    assert o.mask[0, 1] and not o.mask[1, 1]               # row 1 has no ROI -> masked
    assert not o.mask[:, 4].any()                          # knowledge slot masked (ontology not implemented)
    assert o.modality_status["knowledge"] == "not_implemented"
    assert (o.tokens[~o.mask] == 0).all() and torch.isfinite(o.tokens).all()


def test_fusion_runs_without_roi_and_with_patches():
    f = _fusion(use_patch_tokens=True)
    i = _inputs(with_roi=False)
    i.patch_features = torch.randn(2, 9, 24)
    o = f(i)
    assert o.tokens.shape[1] == 5 + 7 + 9 and o.modality_status["roi"] == "missing"


def test_fusion_variable_question_length_mask():
    f = _fusion()
    i = _inputs()
    i.question_mask[1, 4:] = False
    o = f(i)
    assert o.mask[1, 5 + 4:].sum() == 0 and torch.isfinite(o.tokens).all()


def test_localizer_schema_and_roi():
    loc = LesionLocalizer(24, 16)
    p = torch.randn(2, 16, 24)
    out = loc(p, (4, 4))
    assert out["heatmap_logits"].shape == (2, 16) and out["box_xyxy_norm"].shape == (2, 4)
    b = out["box_xyxy_norm"]
    assert ((b >= 0) & (b <= 1)).all()
    box = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    m = box_to_patch_mask(box, (4, 4))
    assert m[0].sum() == 4
    roi, valid = RoIFeatureExtractor()(p, (4, 4), box, torch.tensor([True, False]))
    assert roi.shape == (2, 24) and valid.tolist() == [True, False] and (roi[1] == 0).all()
