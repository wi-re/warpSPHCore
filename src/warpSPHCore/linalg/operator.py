"""Small, explicit matrix-free operator adapter; ordinary callables still work."""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Optional

import torch

__all__ = ['LinearOperator']


@dataclass(frozen=True)
class LinearOperator:
    """A fixed real linear action with optional supplied-output and adjoint actions.

    ``shape`` is (output length, input length). Callbacks must preserve inputs,
    use the current Torch stream, and implement the same fixed action throughout
    a solve. ``matvecOut(v, out)`` writes every output entry and returns None or
    that exact object. The adapter owns no coefficient or execution cache.
    """

    shape: tuple
    dtype: torch.dtype
    device: torch.device
    matvec: Optional[Callable] = None
    matvecOut: Optional[Callable] = None
    adjoint: Optional[Callable] = None

    def __post_init__(self):
        if (not isinstance(self.shape, (tuple, list)) or len(self.shape) != 2
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
                                       for n in self.shape)):
            raise ValueError('shape must contain two nonnegative integer dimensions')
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError('LinearOperator supports float32 and float64')
        for name in ('matvec', 'matvecOut', 'adjoint'):
            action = getattr(self, name)
            if action is not None and not callable(action):
                raise ValueError(f'{name} must be callable')
        if self.matvec is None and self.matvecOut is None:
            raise ValueError('matvec or matvecOut is required')
        device = torch.device(self.device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        object.__setattr__(self, 'device', device)
        object.__setattr__(self, 'shape', tuple(int(n) for n in self.shape))

    def _validate(self, v, length, name):
        if not isinstance(v, torch.Tensor) or v.shape != (length,):
            raise ValueError(f'{name} must have shape ({length},)')
        if v.dtype != self.dtype or v.device != self.device:
            raise ValueError(f'{name} must have dtype {self.dtype} and device {self.device}')
        return v

    def __call__(self, v):
        self._validate(v, self.shape[1], 'operator input')
        if self.matvec is not None:
            return self._validate(self.matvec(v), self.shape[0], 'operator output')
        return self.apply(v, torch.empty(self.shape[0], dtype=self.dtype, device=self.device))

    def apply(self, v, out=None):
        """Apply into optional caller storage, without permitting input/output aliasing."""
        self._validate(v, self.shape[1], 'operator input')
        if out is None:
            return self(v)
        self._validate(out, self.shape[0], 'operator output')
        if torch._C._overlaps(v, out):
            raise ValueError('operator input and supplied output must not overlap')
        if self.matvecOut is None:
            out.copy_(self(v))
        else:
            metadata = (out.data_ptr(), out.shape, out.stride(), out.dtype, out.device)
            returned = self.matvecOut(v, out)
            if returned is not None and returned is not out:
                raise ValueError('matvecOut must return None or the supplied output')
            if metadata != (out.data_ptr(), out.shape, out.stride(), out.dtype, out.device):
                raise ValueError('matvecOut must preserve supplied output storage and metadata')
        return out

    def applyAdjoint(self, v):
        """Apply the explicitly supplied transpose; never infer one from a callback."""
        if self.adjoint is None:
            raise ValueError('operator has no adjoint action')
        self._validate(v, self.shape[0], 'adjoint input')
        return self._validate(self.adjoint(v), self.shape[1], 'adjoint output')
