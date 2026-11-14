"""Điểm DUY NHẤT quyết định Offline hay Online về gradient.

  TRAIN : module.train(), bật gradient          -> dùng trong bone_cvqa.training
  INFER : module.eval(),  torch.inference_mode  -> dùng trong bone_cvqa.inference

Các module (encoder, fusion, decoder...) KHÔNG tự bật/tắt gradient; chúng chỉ chạy forward.
"""
from contextlib import contextmanager

import torch
from torch import nn

from .constants import ExecutionMode


@contextmanager
def execution_context(module: nn.Module, mode: ExecutionMode):
    if mode == ExecutionMode.TRAIN:
        module.train()
        with torch.enable_grad():
            yield
    else:
        module.eval()
        with torch.inference_mode():
            yield
