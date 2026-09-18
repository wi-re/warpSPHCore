import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...type_config import *
from typing import Any
import warp as wp
from warp.types import vector

# The D&A 2012 eq.-11 B-spline family in support-1 units,
#   b_n(q) = sum_i (-1)^i C(n,i) ((n-2i)/n - q)_+^(n-1),
# member n = 8 (order 8, degree 7; named by degree, D&A index by order):
#   b_8(q) = (1-q)^7 - 8(3/4-q)_+^7 + 28(1/2-q)_+^7 - 56(1/4-q)_+^7
# (bpow(x, p) = (min(x, 0))^p, so bpow(q-t, odd) = -(t-q)_+^odd).
# kernelScale is shape-derived (H/h = 1/(2 sqrt(sigma2/H2)), D&A eq. 8,
# sigma2/H2 = 1/24, 531453/13085600, 19/480); previously it copied the
# quintic spline's row, which was off by ~13-15 % (fixed 2026-09-18).

@wp.func
def B8_k(q: scalar_t, dim: wp.int32 = 2):
    return scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 7) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 7) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 7) - bpow_warp(q - scalar_t(1.0), 7)

@wp.func
def B8_dkdq(q: scalar_t, dim: wp.int32 = 2):
    return (scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 6) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 6) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 6) - bpow_warp(q - scalar_t(1.0), 6)) * scalar_t(7.0)

@wp.func
def B8_d2kdq2(q: scalar_t, dim: wp.int32 = 2):
    return (scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 5) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 5) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 5) - bpow_warp(q - scalar_t(1.0), 5)) * scalar_t(42.0)

@wp.func
def B8_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    return (scalar_t(56.0) * bpow_warp(q - scalar_t(0.25), 4) - scalar_t(28.0) * bpow_warp(q - scalar_t(0.5), 4) + scalar_t(8.0) * bpow_warp(q - scalar_t(0.75), 4) - bpow_warp(q - scalar_t(1.0), 4)) * scalar_t(210.0)

@wp.func
def B8_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(4096.0)/scalar_t(315.0)
    elif dim == 2: return scalar_t(589824.0) / (scalar_t(7435.0) * scalar_t(np.pi))
    else: return scalar_t(16384.0) / (scalar_t(105.0) * scalar_t(np.pi))

@wp.func # shape-derived: H/h = 1/(2 sqrt(sigma2/H2)), sigma2/H2 = 1/24, 531453/13085600, 19/480
def B8_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(2.449490)
    elif dim == 2: return scalar_t(2.481044)
    else: return scalar_t(2.513123)

@wp.func # no independent reference (not in D&A Table 2); same convention as the other code B-splines
def B8_packingRatio():
    return scalar_t(1.595) * scalar_t(1.1425) # Factor to be in line CRKSPH
