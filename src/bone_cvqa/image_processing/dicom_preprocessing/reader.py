"""Đọc ảnh X-quang: DICOM (.dcm) hoặc PNG/JPEG -> mảng gray float32 + thông tin cửa sổ (nếu có)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ...commons.exceptions import InvalidImageError

DICOM_SUFFIXES = {".dcm", ".dicom"}


@dataclass
class RawImage:
    pixels: np.ndarray                      # [H, W] float32, MONOCHROME1 đã đảo, đã áp RescaleSlope/Intercept
    window_center: Optional[float] = None   # từ header DICOM (nếu có)
    window_width: Optional[float] = None
    source_format: str = "image"


def _first_value(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value[0]) if hasattr(value, "__len__") and not isinstance(value, (str, bytes)) else float(value)
    except (TypeError, ValueError):
        return None


def read_dicom(path: Path) -> RawImage:
    import pydicom
    try:
        dataset = pydicom.dcmread(str(path))
        pixels = np.asarray(dataset.pixel_array).astype(np.float32)
    except Exception as error:
        raise InvalidImageError(f"Không đọc được DICOM {path}: {error}") from error
    if pixels.ndim == 4:
        pixels = pixels[0]
    if pixels.ndim == 3:   # RGB hoặc nhiều frame
        pixels = pixels[..., :3].mean(axis=-1) if pixels.shape[-1] in (3, 4) else pixels[0]
    if pixels.ndim != 2 or pixels.size == 0:
        raise InvalidImageError(f"Mảng pixel DICOM có shape bất thường {pixels.shape}: {path}")
    pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0) or 1.0) + float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels      # đảo để xương luôn sáng như MONOCHROME2
    if not np.isfinite(pixels).all():
        raise InvalidImageError(f"DICOM có giá trị không hữu hạn: {path}")
    return RawImage(pixels, _first_value(getattr(dataset, "WindowCenter", None)),
                    _first_value(getattr(dataset, "WindowWidth", None)), "dicom")


def read_image(path) -> RawImage:
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"Không tìm thấy file ảnh: {path}")
    if path.suffix.lower() in DICOM_SUFFIXES:
        return read_dicom(path)
    try:
        with Image.open(path) as image:
            image.load()
            sixteen_bit = image.mode in ("I;16", "I;16B", "I;16L", "I")
            pixels = np.asarray(image if sixteen_bit else image.convert("L")).astype(np.float32)
    except Exception as error:
        raise InvalidImageError(f"Không giải mã được ảnh {path}: {error}") from error
    if pixels.ndim != 2 or pixels.size == 0:
        raise InvalidImageError(f"Ảnh có shape bất thường {pixels.shape}: {path}")
    return RawImage(pixels, source_format=path.suffix.lower().lstrip("."))
