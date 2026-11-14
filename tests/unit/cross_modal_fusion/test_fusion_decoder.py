import torch


def test_fusion_and_teacher_forcing(tiny_system, tiny_cfg):
    system = tiny_system
    images = torch.rand(2, 3, 64, 64)
    context = system.encode(images, ["Is there a fracture?", "Where?"], [[], []])
    assert context.fusion.tokens.shape[0] == 2
    assert context.fusion.modality_status["knowledge"] == "not_implemented"
    logits, targets = system.answer_logits(context, ["Yes", "No"])
    assert logits.shape[0] == targets.shape[0] > 0


def test_generate(tiny_system):
    context = tiny_system.encode(torch.rand(1, 3, 64, 64), ["Is there a fracture?"], [[]])
    texts = tiny_system.generate_answers(context, {"max_new_tokens": 4})
    assert len(texts) == 1 and isinstance(texts[0], str)
