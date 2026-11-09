import json
from pathlib import Path

import pytest
import torch

from src.common.config import load_config
from src.data.collate import cvqa_collate
from src.data.dataset_factory import build_dataset
from src.data.image_preprocessing import ImagePreprocessor
from src.models.cvqa_inputs import build_model_inputs
from src.models.model_io import build_cvqa_model, load_model_checkpoint
from src.offline.losses import JointLoss
from src.offline.pipeline import OfflinePipeline
from src.ontology.branch import OntologyBranch
from src.question_processing.processor import QuestionProcessor
from tests.conftest import tiny_cfg


def _batch(data_cfg, n=4):
    ds = build_dataset(data_cfg, "train", ImagePreprocessor({"image_size": 64}))
    return cvqa_collate([ds[i] for i in range(min(n, len(ds)))])


def _forward(model, batch, **kw):
    inp = build_model_inputs(model, QuestionProcessor(), OntologyBranch(), batch["question"], batch["conversation_history"])
    return model(batch["image"], inp["conv_texts"], inp["type_ids"], inp["prompts"], answers=batch["answer"],
                 gt_box=batch["gt_box"], has_box=batch["has_box"], use_gt_roi=True, knowledge=inp["knowledge"], **kw)


def test_offline_forward_pass(tiny_model, data_cfg):
    b = _batch(data_cfg)
    out = _forward(tiny_model, b)
    assert out["answer_logits"].shape[1] == tiny_model.decoder.llm.config.vocab_size
    assert out["fusion_status"]["knowledge"] == "not_implemented"           # ontology branch skipped, not faked
    assert out["loc"]["box_xyxy_norm"].shape == (len(b["question"]), 4)


def test_joint_loss_active_and_skipped(tiny_model, data_cfg):
    b = _batch(data_cfg, 8)
    res = JointLoss({"ans": 1.0, "loc": 0.5})(_forward(tiny_model, b), b)
    assert "ans" in res["active"] and res["total"] is not None
    for k in ("rat", "kg", "align"):
        assert k in res["skipped"] and k not in res["components"]            # never constants
    if b["has_box"].any():
        assert "loc" in res["active"]


def test_loss_without_boxes_skips_loc(tiny_model, data_cfg):
    b = _batch(data_cfg, 4)
    b["has_box"][:] = False
    res = JointLoss()(_forward(tiny_model, b), b)
    assert "loc" in res["skipped"] and "loc" not in res["active"]


def test_loss_with_no_answers_has_no_total(tiny_model, data_cfg):
    b = _batch(data_cfg, 2)
    b["has_box"][:] = False
    inp = build_model_inputs(tiny_model, QuestionProcessor(), OntologyBranch(), b["question"], b["conversation_history"])
    out = tiny_model(b["image"], inp["conv_texts"], inp["type_ids"], inp["prompts"], answers=None, knowledge=inp["knowledge"])
    assert JointLoss()(out, b)["total"] is None


def test_one_training_step_updates_only_trainable(tiny_model, data_cfg):
    b = _batch(data_cfg, 4)
    frozen_before = {n: p.clone() for n, p in tiny_model.named_parameters() if not p.requires_grad}
    from src.offline.optimizer import build_optimizer
    opt = build_optimizer(tiny_model, {"lr": 1e-2})
    tiny_model.train()
    res = JointLoss({"ans": 1.0, "loc": 0.5})(_forward(tiny_model, b), b)
    w0 = tiny_model.adapter.to_llm.weight.clone()
    res["total"].backward(); opt.step()
    assert not torch.equal(w0, tiny_model.adapter.to_llm.weight)
    for n, p in tiny_model.named_parameters():
        if n in frozen_before:
            assert torch.equal(frozen_before[n], p), f"frozen param {n} changed"


def test_checkpoint_roundtrip(tiny_model, tiny_model_cfg, tmp_path):
    from src.models.model_io import save_model_checkpoint
    tiny_model.localizer_trained = True
    save_model_checkpoint(tmp_path / "m.pt", tiny_model, tiny_model_cfg, {"epoch": 3})
    m2, ck = load_model_checkpoint(tmp_path / "m.pt")
    assert ck["epoch"] == 3 and m2.localizer_trained is True
    a = {n: p for n, p in tiny_model.named_parameters() if p.requires_grad}
    for n, p in m2.named_parameters():
        if n in a:
            assert torch.equal(a[n].cpu(), p.cpu())


def test_end_to_end_training_and_eval(tmp_path, fake_fracatlas):
    cfg = tiny_cfg()
    cfg.update({"data": {"name": "fracatlas", "root": str(fake_fracatlas), "split": {"ratios": [0.6, 0.2, 0.2], "seed": 3},
                         "qa": {"source": "weak_template"}},
                "train": {"epochs": 2, "batch_size": 4, "grad_accum_steps": 2, "lr": 1e-3, "min_loc_steps": 1,
                          "amp": False, "log_every": 1000, "max_train_batches": 3, "max_val_batches": 2},
                "losses": {"weights": {"ans": 1.0, "loc": 0.5}},
                "seed": 0, "device": "cpu",
                "output": {"output_dir": str(tmp_path / "out"), "checkpoint_dir": str(tmp_path / "ck"), "log_dir": str(tmp_path / "lg")}})
    pipe = OfflinePipeline(cfg)
    summary = pipe.train()
    assert (tmp_path / "ck" / "best.pt").exists() and (tmp_path / "ck" / "last.pt").exists()
    assert (tmp_path / "ck" / "config.yaml").exists() and (tmp_path / "out" / "split_metadata.json").exists()
    assert "ans" in summary["active_losses"] and summary["localizer_trained"] is True
    ck = torch.load(tmp_path / "ck" / "best.pt", weights_only=False)
    for k in ("optimizer_state", "scheduler_state", "epoch", "best_metric", "model_config", "split_info", "train_config", "active_losses"):
        assert k in ck
    res = pipe.evaluate(str(tmp_path / "ck" / "best.pt"), "test")
    assert res["ontology"]["status"] == "not_implemented" and res["uncertainty"]["status"] == "not_implemented"
    assert res["answer"]["rouge"] == "not_computed"
    assert (tmp_path / "out" / "eval_test.json").exists() and (tmp_path / "out" / "predictions_test.csv").exists()


def test_yaml_configs_load():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "offline.yaml")
    assert cfg["biomedclip"]["allow_download"] is False and cfg["data"]["name"] == "fracatlas"
