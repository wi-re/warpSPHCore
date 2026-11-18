"""Wendland host metadata with the existing Python scalar arithmetic boundaries.

Lattice calibration historically calls reference leaf functions on the host.
A typed operand rounds its operation at the configured precision and returns a
plain Python float; subsequent plain-float products use Python arithmetic.
Those boundaries differ from a compiled float64 radial kernel. Keep them here
for C2/C4/C6 calibration without importing any kernel DSL or changing device
kernel arithmetic. Leaf expressions and iPow/cpow match kernelFunctions and
math/wp_pow; host parity checks allow one float32 step for host power rounding.
"""
import numpy as np
import torch
from ..type_config import get_torch_precision

def _dtype():
    return {torch.float16:np.float16, torch.float32:np.float32, torch.float64:np.float64}[get_torch_precision()]

class _Scalar:
    def __init__(self, value): self.value = float(value)
    def __float__(self): return self.value
    def _binary(self, other, op, reverse=False):
        dtype = _dtype()
        a, b = dtype(self.value), dtype(float(other))
        return float(op(b, a) if reverse else op(a, b))
    def __add__(self, other): return self._binary(other, lambda a,b:a+b)
    def __radd__(self, other): return self._binary(other, lambda a,b:a+b, True)
    def __sub__(self, other): return self._binary(other, lambda a,b:a-b)
    def __rsub__(self, other): return self._binary(other, lambda a,b:a-b, True)
    def __mul__(self, other): return self._binary(other, lambda a,b:a*b)
    def __rmul__(self, other): return self._binary(other, lambda a,b:a*b, True)
    def __truediv__(self, other): return self._binary(other, lambda a,b:a/b)
    def __rtruediv__(self, other): return self._binary(other, lambda a,b:a/b, True)
    def __pow__(self, other): return self._binary(other, lambda a,b:np.power(a,b))
    def __rpow__(self, other): return self._binary(other, lambda a,b:np.power(a,b), True)
    def __neg__(self): return float(-_dtype()(self.value))
    def __lt__(self, other): return self._binary(other,lambda a,b:a<b)
    def __le__(self, other): return self._binary(other,lambda a,b:a<=b)
    def __gt__(self, other): return self._binary(other,lambda a,b:a>b)
    def __ge__(self, other): return self._binary(other,lambda a,b:a>=b)

def _clamp(q, lo, hi):
    dtype = _dtype() if any(isinstance(v,_Scalar) for v in (q,lo,hi)) else np.float32
    return float(min(max(dtype(float(q)),dtype(float(lo))),dtype(float(hi))))

scalar_t = _Scalar


def iPow(base, exp):
    if exp == 0:
        return scalar_t(1.0)
    elif exp == 1:
        return base
    elif exp == 2:
        return base * base
    elif exp == 3:
        return base * base * base
    elif exp == 4:
        return base * base * base * base
    elif exp == 5:
        return base * base * base * base * base
    elif exp == 6:
        return base * base * base * base * base * base
    else:
        return base ** scalar_t(exp)
    return scalar_t(np.nan)

def cpow_warp(q, p):
    return iPow(_clamp(q, scalar_t(0.0), scalar_t(1.0)), p)

def wendland2_k(q, dim=2):
    if dim == 1:
        return cpow_warp(scalar_t(1.0) - q, 3) * (scalar_t(1.0) + scalar_t(3.0) * q)
    else:
        return cpow_warp(scalar_t(1.0) - q, 4) * (scalar_t(1.0) + scalar_t(4.0) * q)

def wendland2_C_d(dim):
    if dim == 1:
        return scalar_t(5.0) / scalar_t(4.0)
    elif dim == 2:
        return scalar_t(7.0) / scalar_t(np.pi)
    else:
        return scalar_t(21.0) / (scalar_t(2.0) * scalar_t(np.pi))

def wendland4_k(q, dim=2):
    if dim == 1:
        return cpow_warp(scalar_t(1.0) - q, 5) * (scalar_t(1.0) + scalar_t(5.0) * q + scalar_t(8.0) * iPow(q, 2))
    else:
        return cpow_warp(scalar_t(1.0) - q, 6) * (scalar_t(1.0) + scalar_t(6.0) * q + scalar_t(35.0) / scalar_t(3.0) * iPow(q, 2))

def wendland4_C_d(dim):
    if dim == 1:
        return scalar_t(3.0) / scalar_t(2.0)
    elif dim == 2:
        return scalar_t(9.0) / scalar_t(np.pi)
    else:
        return scalar_t(495.0) / (scalar_t(32.0) * scalar_t(np.pi))

def wendland6_k(q, dim=2):
    if dim == 1:
        return cpow_warp(scalar_t(1.0) - q, 7) * (scalar_t(1.0) + scalar_t(7.0) * q + scalar_t(19.0) * iPow(q, 2) + scalar_t(21.0) * iPow(q, 3))
    else:
        return cpow_warp(scalar_t(1.0) - q, 8) * (scalar_t(1.0) + scalar_t(8.0) * q + scalar_t(25.0) * iPow(q, 2) + scalar_t(32.0) * iPow(q, 3))

def wendland6_C_d(dim):
    if dim == 1:
        return scalar_t(55.0) / scalar_t(32.0)
    elif dim == 2:
        return scalar_t(78.0) / (scalar_t(7.0) * scalar_t(np.pi))
    else:
        return scalar_t(1365.0) / (scalar_t(64.0) * scalar_t(np.pi))


_RADIAL = (wendland2_k, wendland4_k, wendland6_k)
_COEFFICIENT = (wendland2_C_d, wendland4_C_d, wendland6_C_d)

def radial(q, dim, kernel):
    return float(_RADIAL[kernel](_Scalar(q), dim))

def coefficient(dim, kernel):
    return float(_COEFFICIENT[kernel](dim))
