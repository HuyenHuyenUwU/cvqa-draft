import torch

from src.models.tiny_factory import TinyBiomedCLIP, DummyClipTokenizer
from src.models.visual.biomedclip_encoder import BiomedCLIPVisualEncoder
from src.models.language.text_encoder import ConversationTextEncoder


def test_visual_encoder_shapes_and_equivalence():
    clip = TinyBiomedCLIP(embed_dim=32, image_size=64)
    enc = BiomedCLIPVisualEncoder(clip.visual, 64, freeze=True)
    x = torch.randn(2, 3, 64, 64)
    out = enc(x)
    assert out.patch_features.shape == (2, 16, enc.patch_dim) and out.grid_size == (4, 4)
    assert out.global_embedding.shape == (2, 32) == (2, enc.global_dim)
    # our extraction must equal open_clip's own visual(x) for the global embedding
    assert torch.allclose(out.global_embedding, clip.visual.eval()(x), atol=1e-5)
    assert not any(p.requires_grad for p in enc.parameters())


def test_visual_freeze_unfreeze():
    clip = TinyBiomedCLIP()
    enc = BiomedCLIPVisualEncoder(clip.visual, 64, freeze=False)
    assert all(p.requires_grad for p in enc.parameters())
    enc.set_trainable(True, unfreeze_last_n_blocks=1)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert trainable and all(("blocks.%d" % (len(enc.trunk.blocks) - 1)) in n or "norm" in n or "head" in n
                             for n in trainable)


def test_text_encoder_shapes_mask_and_history():
    clip = TinyBiomedCLIP()
    enc = ConversationTextEncoder("biomedclip", clip.text, DummyClipTokenizer(), max_length=96, freeze=True)
    texts = ["[CURRENT QUESTION]\nis there a fracture", "[HISTORY]\nUser: a b c d e\nAssistant: x y z\n[CURRENT QUESTION]\nwhere"]
    o = enc(texts)
    B, T, Dt = o.question_features.shape
    assert B == 2 and Dt == enc.feature_dim and o.question_embedding.shape == (2, enc.embedding_dim)
    assert o.question_mask.shape == (B, T) and o.question_mask.dtype == torch.bool
    assert o.question_mask[0].sum() < o.question_mask[1].sum()          # shorter text -> fewer valid tokens
    # padding must not change the pooled embedding of a given text
    solo = enc([texts[0]])
    assert torch.allclose(solo.question_embedding[0], o.question_embedding[0], atol=1e-5)


def test_text_encoder_empty_history_ok():
    clip = TinyBiomedCLIP()
    enc = ConversationTextEncoder("biomedclip", clip.text, DummyClipTokenizer(), freeze=True)
    assert enc(["[CURRENT QUESTION]\nhi"]).question_embedding.shape[0] == 1
