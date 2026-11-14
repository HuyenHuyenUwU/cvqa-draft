import subprocess
import sys
from pathlib import Path

import torch

from bone_cvqa.commons.config import deep_merge
from bone_cvqa.inference.pipeline import OnlinePipeline

ROOT = Path(__file__).resolve().parents[2]


def test_online_no_grad_and_params_unchanged(tiny_system, tiny_cfg, xray_path):
    before = {k: v.clone() for k, v in tiny_system.state_dict().items()}
    pipeline = OnlinePipeline(tiny_system, tiny_cfg)
    result = pipeline.run(xray_path, "Is there a fracture in this X-ray image?", [], "s1")
    assert result["pipeline_status"] == "completed"
    after = tiny_system.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in tiny_system.parameters())


def test_out_of_scope_skips_generation(tiny_system, tiny_cfg, xray_path):
    result = OnlinePipeline(tiny_system, tiny_cfg).run(xray_path, "What is the capital of France?", [], "s2")
    assert result["pipeline_status"] == "completed_out_of_scope" and result["answer"] is None


def _modules_after(import_stmt):
    code = f"import sys; {import_stmt}; print(sorted(m for m in sys.modules if m.startswith('bone_cvqa')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}).stdout
    return out


def test_offline_does_not_import_online_only():
    out = _modules_after("import bone_cvqa.training.pipeline")
    for name in ("image_quality", "scope_detector", "uncertainty_estimation", "trustworthy_verification", "bone_cvqa.inference"):
        assert name not in out, name


def test_online_does_not_import_training():
    out = _modules_after("import bone_cvqa.inference.engine")
    assert "bone_cvqa.training" not in out


def test_end2end_script(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_end2end.py"), "--workdir", str(tmp_path / "e2e")],
                            capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert "END-TO-END OK" in result.stdout, result.stderr[-2000:]
