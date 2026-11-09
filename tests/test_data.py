import numpy as np
import pytest
import torch

from src.common.exceptions import DatasetError, DicomError, InvalidImageError
from src.common.schemas import ANSWER_SOURCE_WEAK
from src.data.collate import cvqa_collate
from src.data.dataset_factory import build_dataset
from src.data.image_preprocessing import ImagePreprocessor
from src.data.splits import assign_splits
from src.data.text_processing import build_conversation_text

PRE = ImagePreprocessor({"image_size": 64})


def test_conversation_text_empty_history():
    t = build_conversation_text("Is there a fracture?", [])
    assert t.startswith("[CURRENT QUESTION]") and "[HISTORY]" not in t


def test_conversation_text_order_and_formats():
    h = [{"question": "q1", "answer": "a1"}, ("q2", "a2"), {"role": "user", "content": "q3"}]
    t = build_conversation_text("now?", h)
    assert t.index("q1") < t.index("a1") < t.index("q2") < t.index("q3") < t.index("[CURRENT QUESTION]")
    assert "User: q1" in t and "Assistant: a1" in t


def test_gray_and_rgb_preprocessing():
    gray = (np.random.rand(50, 70) * 255).astype(np.float32)
    rgb = (np.random.rand(50, 70, 3) * 255).astype(np.float32)
    a, b = PRE(gray), PRE(rgb)
    assert a.tensor.shape == (3, 64, 64) == b.tensor.shape
    assert a.original_size == (50, 70) and 0 <= a.gray.min() and a.gray.max() <= 1


def test_constant_image_is_flat_not_crash():
    out = PRE(np.full((40, 40), 7.0, dtype=np.float32))
    assert out.gray.max() == 0.0


def test_invalid_file(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        PRE(str(bad))
    with pytest.raises(InvalidImageError):
        PRE(str(tmp_path / "missing.png"))


def _write_dicom(path, arr, photometric="MONOCHROME2"):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelData = arr.astype(np.uint16).tobytes()
    pydicom.dcmwrite(str(path), ds, enforce_file_format=True)


def test_dicom_monochrome1_inverted(tmp_path):
    arr = np.tile(np.linspace(0, 4000, 40, dtype=np.uint16), (40, 1))
    _write_dicom(tmp_path / "m2.dcm", arr, "MONOCHROME2")
    _write_dicom(tmp_path / "m1.dcm", arr, "MONOCHROME1")
    from src.data.dicom_preprocessing import load_dicom_gray
    m2, m1 = load_dicom_gray(tmp_path / "m2.dcm"), load_dicom_gray(tmp_path / "m1.dcm")
    assert m2.shape == (40, 40) and m2[0, -1] > m2[0, 0]
    assert m1[0, -1] < m1[0, 0]                      # inverted
    assert PRE(str(tmp_path / "m1.dcm")).tensor.shape == (3, 64, 64)


def test_corrupted_dicom(tmp_path):
    p = tmp_path / "broken.dcm"
    p.write_bytes(b"\0" * 300)
    with pytest.raises(DicomError):
        PRE(str(p))


def test_fracatlas_adapter(data_cfg, tmp_path):
    ds = build_dataset(data_cfg, "train", PRE, str(tmp_path / "split.json"))
    assert len(ds) > 0
    s = ds[0]
    assert set(["sample_id", "image", "question", "conversation_history", "answer",
                "lesion_annotation", "metadata"]) <= set(s)
    assert s["metadata"]["answer_source"] == ANSWER_SOURCE_WEAK       # never labelled expert
    assert (tmp_path / "split.json").exists()


def test_split_deterministic_and_disjoint(data_cfg):
    a = {sp: build_dataset(data_cfg, sp, PRE) for sp in ("train", "val", "test")}
    ids = {sp: {r["image_id"] for r in d.records} for sp, d in a.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    b = build_dataset(data_cfg, "train", PRE)
    assert {r["image_id"] for r in b.records} == ids["train"]


def test_patient_level_split():
    recs = [{"image_id": f"i{k}", "patient_id": f"p{k % 5}"} for k in range(40)]
    sp = assign_splits(recs, seed=3)
    for p in range(5):
        assert len({sp[f"i{k}"] for k in range(40) if k % 5 == p}) == 1


def test_missing_annotations_marked_missing(data_cfg):
    ds = build_dataset(data_cfg, "train", PRE)
    batch = cvqa_collate([ds[i] for i in range(len(ds))])
    for s, hb, m in zip(range(len(ds)), batch["has_box"], batch["metadata"]):
        assert bool(hb) == (m["label"] == 1 and ds.items[s]["lesion"]["status"] == "available")
    assert (batch["gt_box"][~batch["has_box"]] == 0).all()           # explicit mask, not a target
    assert batch["image"].shape[1:] == (3, 64, 64)


def test_history_grows_with_turns(data_cfg):
    ds = build_dataset(data_cfg, "train", PRE)
    two = [i for i, it in enumerate(ds.items) if it["turn"] == 1]
    assert two and len(ds[two[0]]["conversation_history"]) == 1


def test_missing_root():
    with pytest.raises(DatasetError):
        build_dataset({"name": "fracatlas", "root": "/nonexistent"}, "train", PRE)
