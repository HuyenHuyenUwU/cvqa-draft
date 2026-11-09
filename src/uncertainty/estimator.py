from __future__ import annotations

from ..common.logger import get_logger, log_placeholder
from ..common.schemas import Status
from .interfaces import ConfidenceEstimator, UncertaintyEstimator

log = get_logger("uncertainty")


class PlaceholderUncertaintyEstimator(UncertaintyEstimator):
    def __init__(self):
        self._calls = 0

    def estimate(self, logits=None, multimodal_repr=None, answer=None):
        if self._calls == 0:
            log_placeholder(log, "Uncertainty Estimation")
        self._calls += 1
        return {"score": None, "status": Status.NOT_IMPLEMENTED}


class PlaceholderConfidenceEstimator(ConfidenceEstimator):
    def __init__(self):
        self._calls = 0

    def estimate(self, uncertainty=None):
        if self._calls == 0:
            log_placeholder(log, "Confidence Score")
        self._calls += 1
        return {"score": None, "status": Status.NOT_IMPLEMENTED}
