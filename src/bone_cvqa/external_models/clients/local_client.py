"""Client nạp model HuggingFace chạy LOCAL. Không tải gì nếu allow_download=false."""
from pathlib import Path
from typing import Any, Dict

from ...commons.exceptions import ModelLoadError


def hf_source(config: Dict[str, Any]) -> Dict[str, Any]:
    """Trả về kwargs `pretrained_model_name_or_path` + `local_files_only` theo chính sách tải."""
    name = config.get("local_dir") or config.get("name")
    if not name:
        raise ModelLoadError("Thiếu `name` hoặc `local_dir` trong cấu hình model.")
    is_local_dir = Path(str(name)).is_dir()
    allow_download = bool(config.get("allow_download", False))
    return {"pretrained_model_name_or_path": str(name),
            "local_files_only": is_local_dir or not allow_download}


def load_hf_model(auto_class: Any, config: Dict[str, Any], **kwargs: Any):
    source = hf_source(config)
    try:
        return auto_class.from_pretrained(**source, **kwargs)
    except OSError as error:
        raise ModelLoadError(
            f"Không nạp được '{source['pretrained_model_name_or_path']}' (local_files_only="
            f"{source['local_files_only']}). Đặt local_dir tới bản đã tải, hoặc allow_download: true. ({error})") from error
