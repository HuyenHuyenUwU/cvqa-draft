"""Các exception tường minh: lỗi dữ liệu/model không bao giờ bị nuốt im lặng."""


class BoneCVQAError(Exception):
    """Lớp gốc của mọi lỗi trong dự án."""


class ConfigError(BoneCVQAError):
    pass


class DatasetError(BoneCVQAError):
    pass


class InvalidImageError(BoneCVQAError):
    """Ảnh hỏng / rỗng / không đọc được (kể cả DICOM)."""


class ModelLoadError(BoneCVQAError):
    """Thiếu trọng số, hoặc cần tải nhưng allow_download=false."""


class CheckpointError(BoneCVQAError):
    pass
