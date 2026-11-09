"""Explicit exception types so failures are never silent."""


class CVQAError(Exception):
    """Base class for all project errors."""


class ConfigError(CVQAError):
    pass


class DatasetError(CVQAError):
    pass


class InvalidImageError(CVQAError):
    """Raised for corrupted / empty / unreadable images (incl. DICOM)."""


class DicomError(InvalidImageError):
    pass


class CheckpointError(CVQAError):
    pass


class ModelLoadError(CVQAError):
    """Raised when a pretrained checkpoint is missing or download is not allowed."""
