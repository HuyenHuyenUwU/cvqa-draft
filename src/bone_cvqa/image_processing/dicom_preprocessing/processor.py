"""DICOM Preprocessing (khối 1, dùng chung Offline và Online).

  đường dẫn | mảng numpy -> read -> window -> resize -> 3 kênh -> chuẩn hóa mean/std -> PreprocessedImage
Training và inference dùng CHÍNH lớp này với cùng cấu hình (cfg["image_processing"]["preprocess"]).
Box tổn thương lưu dạng chuẩn hóa theo ảnh gốc nên resize vuông không làm sai box.
"""
from typing import Any, Dict, Optional, Union

import numpy as np

from ...commons.exceptions import InvalidImageError
from ...commons.types import PreprocessedImage
from .normalization import min_max_unit, normalize_mean_std
from .reader import RawImage, read_image
from .transforms import gray_to_three_channels, resize_square
from .windowing import window_to_unit

DEFAULTS = {"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "window_mode": "percentile", "window_percentiles": [0.5, 99.5]}


class DicomPreprocessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = {**DEFAULTS, **(config or {})}
        self.image_size = int(self.config["image_size"])

    def __call__(self, source: Union[str, np.ndarray]) -> PreprocessedImage:
        if isinstance(source, np.ndarray):
            pixels = source.astype(np.float32)
            pixels = pixels.mean(axis=-1) if pixels.ndim == 3 else pixels
            raw = RawImage(pixels)
        else:
            raw = read_image(source)
        if raw.pixels.ndim != 2 or raw.pixels.size == 0:
            raise InvalidImageError(f"Ảnh có shape bất thường {raw.pixels.shape}")
        windowed = window_to_unit(raw, self.config["window_mode"], self.config["window_percentiles"])
        resized = resize_square(windowed, self.image_size)
        tensor = normalize_mean_std(gray_to_three_channels(resized), self.config["mean"], self.config["std"])
        quality_view = resize_square(min_max_unit(raw.pixels), self.image_size)
        return PreprocessedImage(tensor=tensor, gray_unit=quality_view, original_size_hw=tuple(raw.pixels.shape))
