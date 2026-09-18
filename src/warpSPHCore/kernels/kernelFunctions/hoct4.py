import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...type_config import *
from typing import Any
import warp as wp
from warp.types import vector

# HOCT4 kernel (Read et al. 2010, eqs. 46-51; Dehnen & Aly 2012, Table 1
# / Fig. 1 comparison kernel). nk = 4, beta = 0.5, gamma = 0.75; A and B
# exact rationals; alpha = smaller root of 14.6 a^2 - 12 a + 1.9 = 0.
# Verified definition: higherOrderSPH/dehnen2012_.../data/da2012_reference.yaml
# (hoct4_definition; cross-checked against the CT kernel of the same
# paper, normalisation ratio 1.000000). W(r) = N_d f(x)/H^d, x = r/H:
#   f(x) = P x + Q                                     0 <= x <= alpha
#        = (1-x)^4 + A (gamma-x)_+^4 + B (beta-x)_+^4  alpha < x <= 1
# C^2-continuous at alpha/beta/gamma; monotonically decreasing; the
# central cusp f'(0) = P != 0 is by design ("constant central core to
# the kernel gradient" / triangular central spike). The same shape f is
# used in all dims, with the per-dim normalisation N_d.

# Constants as module-level Python floats: a `scalar_t(16.0/5.0)` binop
# inside a traced func is evaluated in float32 by the warp tracer (warp
# gotcha, same class as the B7 knot constants); a plain module constant
# referenced by name survives float64 exactly (verified).
_HOCT4_ALPHA = 0.214108111463
_HOCT4_BETA = 0.5
_HOCT4_GAMMA = 0.75
_HOCT4_A = 16.0 / 5.0   # (1 - beta^2)/(gamma^(nk-3) (gamma^2 - beta^2)), exact
_HOCT4_B = -94.0 / 5.0  # -(1 + A gamma^(nk-1))/beta^(nk-1), exact
_HOCT4_P = -2.154228492776   # eq. 49
_HOCT4_Q = 0.981018558675    # eq. 50
# derivative coefficients (Python-side, full precision)
_HOCT4_4A = 4.0 * _HOCT4_A
_HOCT4_4B = 4.0 * _HOCT4_B
_HOCT4_12A = 12.0 * _HOCT4_A
_HOCT4_12B = 12.0 * _HOCT4_B
_HOCT4_24A = 24.0 * _HOCT4_A
_HOCT4_24B = 24.0 * _HOCT4_B

@wp.func
def HOCT4_k(q: scalar_t, dim: wp.int32 = 3):
    quartic = bpow_warp(q - scalar_t(1.0), 4) + scalar_t(_HOCT4_A) * bpow_warp(q - scalar_t(_HOCT4_GAMMA), 4) + scalar_t(_HOCT4_B) * bpow_warp(q - scalar_t(_HOCT4_BETA), 4)
    linear = scalar_t(_HOCT4_P) * q + scalar_t(_HOCT4_Q)
    return wp.where(q <= scalar_t(_HOCT4_ALPHA), linear, quartic)

# Note: d/dq bpow_warp(q - t, p) = p * bpow_warp(q - t, p - 1) (the sign
# lives in the negative argument, min(q - t, 0) <= 0) -- no leading minus.
@wp.func
def HOCT4_dkdq(q: scalar_t, dim: wp.int32 = 3):
    quartic = scalar_t(4.0) * bpow_warp(q - scalar_t(1.0), 3) + scalar_t(_HOCT4_4A) * bpow_warp(q - scalar_t(_HOCT4_GAMMA), 3) + scalar_t(_HOCT4_4B) * bpow_warp(q - scalar_t(_HOCT4_BETA), 3)
    linear = scalar_t(_HOCT4_P)
    return wp.where(q <= scalar_t(_HOCT4_ALPHA), linear, quartic)

@wp.func
def HOCT4_d2kdq2(q: scalar_t, dim: wp.int32 = 3):
    quartic = scalar_t(12.0) * bpow_warp(q - scalar_t(1.0), 2) + scalar_t(_HOCT4_12A) * bpow_warp(q - scalar_t(_HOCT4_GAMMA), 2) + scalar_t(_HOCT4_12B) * bpow_warp(q - scalar_t(_HOCT4_BETA), 2)
    return wp.where(q <= scalar_t(_HOCT4_ALPHA), scalar_t(0.0), quartic)

@wp.func
def HOCT4_d3kdq3(q: scalar_t, dim: wp.int32 = 3):
    quartic = scalar_t(24.0) * bpow_warp(q - scalar_t(1.0), 1) + scalar_t(_HOCT4_24A) * bpow_warp(q - scalar_t(_HOCT4_GAMMA), 1) + scalar_t(_HOCT4_24B) * bpow_warp(q - scalar_t(_HOCT4_BETA), 1)
    return wp.where(q <= scalar_t(_HOCT4_ALPHA), scalar_t(0.0), quartic)

@wp.func # verified read2010 N_d (3D = the paper's Table-1 10-dp N_3d; 1D/2D from the same verified definition)
def HOCT4_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(2.0684349363)
    elif dim == 2: return scalar_t(3.7158334191)
    else: return scalar_t(6.5150499306)

@wp.func # H/h = 1/(2 sqrt(sigma2/H2)); sigma2/H2 = 0.0505294124, 0.0520086521, 0.0521407118
def HOCT4_kernelScale(dim: wp.int32 = 3):
    if dim == 1: return scalar_t(2.224323)
    elif dim == 2: return scalar_t(2.192463)
    else: return scalar_t(2.189684410)

@wp.func # D&A Table 2: N_H = 442, packing 2.158
def HOCT4_packingRatio():
    return scalar_t(2.158)
