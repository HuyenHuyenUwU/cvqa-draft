"""Cửa sổ cường độ (windowing): đưa pixel về [0, 1].

mode = "header"     : dùng WindowCenter/Width của DICOM nếu có, nếu không thì rơi về percentile
mode = "percentile" : cắt tại hai percentile (mặc định 0.5 - 99.5), ổn định với ảnh JPEG/PNG
Ảnh hằng số (max == min) trả về toàn 0 để Image Quality (online) phát hiện, không chia cho 0.
"""
from typing import Sequence

import numpy as np

from ...commons.exceptions import InvalidImageError
from .reader import RawImage


def window_to_unit(raw: RawImage, mode: str = "percentile", percentiles: Sequence[float] = (0.5, 99.5)) -> np.ndarray:
    pixels = raw.pixels
    if mode == "header" and raw.window_center is not None and raw.window_width and raw.window_width > 1:
        low, high = raw.window_center - raw.window_width / 2, raw.window_center + raw.window_width / 2
    else:
        low, high = np.percentile(pixels, percentiles[0]), np.percentile(pixels, percentiles[1])
    if not (np.isfinite(low) and np.isfinite(high)):
        raise InvalidImageError("Cường độ pixel không hữu hạn")
    if high - low < 1e-6:
        return np.zeros_like(pixels, dtype=np.float32)
    return np.clip((pixels - low) / (high - low), 0.0, 1.0).astype(np.float32)
