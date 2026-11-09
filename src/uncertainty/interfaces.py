"""Uncertainty / confidence interfaces.

UncertaintyEstimator.estimate(logits, multimodal_repr, answer) -> {"score": float|None, "status": str}
ConfidenceEstimator.estimate(uncertainty)                      -> {"score": float|None, "status": str}
Raw token probabilities must NOT be reported as calibrated medical-correctness confidence."""
from abc import ABC, abstractmethod
from typing import Any, Dict


class UncertaintyEstimator(ABC):
    @abstractmethod
    def estimate(self, logits=None, multimodal_repr=None, answer=None) -> Dict[str, Any]: ...


class ConfidenceEstimator(ABC):
    @abstractmethod
    def estimate(self, uncertainty: Dict[str, Any]) -> Dict[str, Any]: ...
