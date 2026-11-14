"""Registry: (role, provider) -> lớp adapter. Đổi công nghệ chỉ cần sửa YAML `external_models`."""
from typing import Any, Callable, Dict, List, Tuple, Type

from ..commons.exceptions import ConfigError
from .base import BaseModelAdapter

_REGISTRY: Dict[Tuple[str, str], Type[BaseModelAdapter]] = {}


def register_adapter(role: str, *providers: str) -> Callable:
    def decorator(adapter_class: Type[BaseModelAdapter]) -> Type[BaseModelAdapter]:
        adapter_class.role = role
        for provider in providers:
            _REGISTRY[(role, provider)] = adapter_class
        return adapter_class
    return decorator


def list_adapters() -> List[Tuple[str, str, str]]:
    return sorted((role, provider, cls.__name__) for (role, provider), cls in _REGISTRY.items())


def build_adapter(role: str, config: Dict[str, Any], auto_load: bool = True) -> BaseModelAdapter:
    provider = config.get("provider")
    key = (role, provider)
    if key not in _REGISTRY:
        available = [p for (r, p) in _REGISTRY if r == role]
        raise ConfigError(f"Không có adapter cho role='{role}', provider='{provider}'. Có sẵn: {available}")
    adapter = _REGISTRY[key](config)
    if auto_load and config.get("enabled", True):
        adapter.load()
    return adapter
