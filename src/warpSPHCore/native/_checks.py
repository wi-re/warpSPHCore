"""Shared execution capability checks; numerical policy belongs to modules."""
import torch
from torch.autograd import forward_ad

def _forward_only(*values):
    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_cuda and torch.cuda.is_current_stream_capturing():
            raise NotImplementedError('Native pilot does not support graph capture')
        if forward_ad.unpack_dual(value).tangent is not None:
            raise NotImplementedError('Native pilot does not support forward-mode AD')
        if torch.is_grad_enabled() and value.requires_grad:
            raise NotImplementedError('Native pilot is forward-only; use the Warp backend for autograd')
