"""Differentiable (torch) kernel values ``W(r, h)`` for the meshless pair
algebra.

The meshless finite-mass / finite-volume geometry (``geometry.py``) only needs
kernel *values* -- never kernel gradients -- evaluated at pair distances for
both particles' supports, so they are written in torch here: the whole MFM
backend is then differentiable in positions and supports (reverse mode and
``torch.autograd.forward_ad``) without a dedicated kernel. Same functions,
normalisations and support convention as ``kernels/kernelFunctions``
(compact support ``r < h``, ``W = C_d k(r/h) / h^dim``); ``tests/operations/
test_mfm.py`` checks the sums against the warp ``Density`` operator.
"""

from __future__ import annotations

import math

import torch

from ..enumTypes import KernelFunctions

__all__ = ["kernelWeight", "kernelDerivative", "SUPPORTED_KERNELS"]


def _c(x: torch.Tensor, p: int) -> torch.Tensor:
    """``cpow_warp``: ``clamp(x, 0, 1) ** p``."""
    return torch.clamp(x, 0.0, 1.0) ** p


def _wendland2(q, dim):
    return _c(1 - q, 3) * (1 + 3 * q) if dim == 1 else _c(1 - q, 4) * (1 + 4 * q)


def _wendland4(q, dim):
    if dim == 1:
        return _c(1 - q, 5) * (1 + 5 * q + 8 * q ** 2)
    return _c(1 - q, 6) * (1 + 6 * q + 35.0 / 3.0 * q ** 2)


def _wendland6(q, dim):
    if dim == 1:
        return _c(1 - q, 7) * (1 + 7 * q + 19 * q ** 2 + 21 * q ** 3)
    return _c(1 - q, 8) * (1 + 8 * q + 25 * q ** 2 + 32 * q ** 3)


def _cubic(q, dim):
    return _c(1 - q, 3) - 4 * _c(0.5 - q, 3)


def _quintic(q, dim):
    return _c(1 - q, 5) - 6 * _c(2.0 / 3.0 - q, 5) + 15 * _c(1.0 / 3.0 - q, 5)


def _d_wendland2(q, dim):
    return -12 * q * _c(1 - q, 2) if dim == 1 else -20 * q * _c(1 - q, 3)


def _d_wendland4(q, dim):
    if dim == 1:
        return -14 * q * (4 * q + 1) * _c(1 - q, 4)
    return -56.0 / 3.0 * q * (5 * q + 1) * _c(1 - q, 5)


def _d_wendland6(q, dim):
    if dim == 1:
        return -6 * q * (35 * q ** 2 + 18 * q + 3) * _c(1 - q, 6)
    return -22 * q * (16 * q ** 2 + 7 * q + 1) * _c(1 - q, 7)


def _d_cubic(q, dim):
    return -3 * _c(1 - q, 2) + 12 * _c(0.5 - q, 2)


def _d_quintic(q, dim):
    return -5 * _c(1 - q, 4) + 30 * _c(2.0 / 3.0 - q, 4) - 75 * _c(1.0 / 3.0 - q, 4)


_PI = math.pi
# kernel -> (shape function, C_d for dim = 1, 2, 3)
_KERNELS = {
    KernelFunctions.Wendland2: (_wendland2, (5 / 4, 7 / _PI, 21 / (2 * _PI))),
    KernelFunctions.Wendland4: (_wendland4, (3 / 2, 9 / _PI, 495 / (32 * _PI))),
    KernelFunctions.Wendland6: (_wendland6, (55 / 32, 78 / (7 * _PI), 1365 / (64 * _PI))),
    KernelFunctions.CubicSpline: (_cubic, (8 / 3, 80 / (7 * _PI), 16 / _PI)),
    KernelFunctions.QuinticSpline: (_quintic, (243 / 40, 15309 / (478 * _PI), 2187 / (40 * _PI))),
}
SUPPORTED_KERNELS = tuple(_KERNELS)
_DERIVATIVES = {
    KernelFunctions.Wendland2: _d_wendland2, KernelFunctions.Wendland4: _d_wendland4,
    KernelFunctions.Wendland6: _d_wendland6, KernelFunctions.CubicSpline: _d_cubic,
    KernelFunctions.QuinticSpline: _d_quintic,
}


def kernelWeight(r: torch.Tensor, h: torch.Tensor, dim: int, kernel: KernelFunctions) -> torch.Tensor:
    """``W(r, h)`` for distances ``r`` and supports ``h`` (broadcastable),
    zero for ``r > h``."""
    if kernel not in _KERNELS:
        raise NotImplementedError(
            f"mfm.kernelWeight supports {[k.name for k in _KERNELS]}, got {kernel.name}")
    shape, C = _KERNELS[kernel]
    q = r / h
    w = shape(q, dim) * (C[dim - 1] / h ** dim)
    return torch.where(q <= 1.0, w, torch.zeros_like(w))


def kernelDerivative(r: torch.Tensor, h: torch.Tensor, dim: int, kernel: KernelFunctions) -> torch.Tensor:
    """``dW/dr (r, h) = C_d k'(q) / h^(dim + 1)`` (non-positive), zero for ``r > h``."""
    if kernel not in _DERIVATIVES:
        raise NotImplementedError(f"mfm.kernelDerivative supports {[k.name for k in _DERIVATIVES]}, got {kernel.name}")
    _, C = _KERNELS[kernel]
    q = r / h
    d = _DERIVATIVES[kernel](q, dim) * (C[dim - 1] / h ** (dim + 1))
    return torch.where(q <= 1.0, d, torch.zeros_like(d))
