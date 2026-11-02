"""Shared execution capability checks; numerical policy belongs to modules."""
import torch
from torch.autograd import forward_ad

def check_execution(*values, allow_reverse=False, allow_forward=False):
    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        if not allow_forward and forward_ad.unpack_dual(value).tangent is not None:
            raise NotImplementedError('Native pilot does not support forward-mode AD')
        if not allow_reverse and torch.is_grad_enabled() and value.requires_grad:
            raise NotImplementedError('Native pilot is forward-only; use the Warp backend for autograd')


def _forward_only(*values):
    check_execution(*values)
