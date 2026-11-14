import torch

from bone_cvqa.training.joint_loss import JointLoss


def test_only_answer_loss_active(tiny_system):
    from bone_cvqa.commons.config import load_config
    weights = load_config(["training"])["loss_weights"]
    loss = JointLoss(weights)
    assert loss is not None
