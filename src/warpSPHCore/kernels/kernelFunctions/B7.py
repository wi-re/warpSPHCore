import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...type_config import *
from typing import Any
import warp as wp
from warp.types import vector

# The D&A 2012 eq.-11 B-spline family in support-1 units,
#   b_n(q) = sum_i (-1)^i C(n,i) ((n-2i)/n - q)_+^(n-1),
# member n = 7 (order 7, degree 6):
#   b_7(q) = (1-q)^6 - 7(5/7-q)_+^6 + 21(3/7-q)_+^6 - 35(1/7-q)_+^6
# (for even powers bpow(q-t, p) = (t-q)_+^p).
# C and kernelScale are shape-derived (normalisation and
# H/h = 1/(2 sqrt(sigma2/H2)), D&A eq. 8 with sigma2/H2 = 1/21,
# 7691281/166329030, 11/245); derived 2026-09-18. The 1D moment is
# exactly on the family pattern sigma2/H2 = 1/(3n) (n = 7 -> 1/21).

# Knots as module-level Python floats: a `scalar_t(5.0/7.0)` binop inside
# a traced func is evaluated in float32 by the warp tracer (warp gotcha,
# same class as the properties.py constant-BinOp issue); a plain module
# constant referenced by name survives float64 exactly (verified).
_B7_KNOT1 = 1.0 / 7.0
_B7_KNOT3 = 3.0 / 7.0
_B7_KNOT5 = 5.0 / 7.0

@wp.func
def B7_k(q: scalar_t, dim: wp.int32 = 2):
    return bpow_warp(q - scalar_t(1.0), 6) - scalar_t(7.0) * bpow_warp(q - scalar_t(_B7_KNOT5), 6) + scalar_t(21.0) * bpow_warp(q - scalar_t(_B7_KNOT3), 6) - scalar_t(35.0) * bpow_warp(q - scalar_t(_B7_KNOT1), 6)

@wp.func
def B7_dkdq(q: scalar_t, dim: wp.int32 = 2):
    return (bpow_warp(q - scalar_t(1.0), 5) - scalar_t(7.0) * bpow_warp(q - scalar_t(_B7_KNOT5), 5) + scalar_t(21.0) * bpow_warp(q - scalar_t(_B7_KNOT3), 5) - scalar_t(35.0) * bpow_warp(q - scalar_t(_B7_KNOT1), 5)) * scalar_t(6.0)

@wp.func
def B7_d2kdq2(q: scalar_t, dim: wp.int32 = 2):
    return (bpow_warp(q - scalar_t(1.0), 4) - scalar_t(7.0) * bpow_warp(q - scalar_t(_B7_KNOT5), 4) + scalar_t(21.0) * bpow_warp(q - scalar_t(_B7_KNOT3), 4) - scalar_t(35.0) * bpow_warp(q - scalar_t(_B7_KNOT1), 4)) * scalar_t(30.0)

@wp.func
def B7_d3kdq3(q: scalar_t, dim: wp.int32 = 2):
    return (bpow_warp(q - scalar_t(1.0), 3) - scalar_t(7.0) * bpow_warp(q - scalar_t(_B7_KNOT5), 3) + scalar_t(21.0) * bpow_warp(q - scalar_t(_B7_KNOT3), 3) - scalar_t(35.0) * bpow_warp(q - scalar_t(_B7_KNOT1), 3)) * scalar_t(120.0)

@wp.func
def B7_C_d(dim: wp.int32):
    if dim == 1: return scalar_t(823543.0) / scalar_t(92160.0)
    elif dim == 2: return scalar_t(5764801.0) / (scalar_t(113149.0) * scalar_t(np.pi))
    else: return scalar_t(5764801.0) / (scalar_t(61440.0) * scalar_t(np.pi))

@wp.func # shape-derived: H/h = 1/(2 sqrt(sigma2/H2)), sigma2/H2 = 1/21, 7691281/166329030, 11/245
def B7_kernelScale(dim: wp.int32 = 2):
    if dim == 1: return scalar_t(2.291288)
    elif dim == 2: return scalar_t(2.325170)
    else: return scalar_t(2.359700)

@wp.func # no independent reference (not in D&A Table 2); same convention as the other code B-splines
def B7_packingRatio():
    return scalar_t(1.595) * scalar_t(1.1425) # Factor to be in line CRKSPH
