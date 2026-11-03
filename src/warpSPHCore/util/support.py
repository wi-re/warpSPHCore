from warp.types import vector, matrix

from warpSPHCore.math.wp_distance import computeDistanceVec
from ..math import safe_sqrt
from typing import Optional, Any, Union, List, Tuple
import math
import numpy as np
import warp as wp

import torch
from ..type_config import scalar_t
from ..enumTypes import SupportScheme
from ..math import vectorNorm_warp
from ..dataTypes.domain_t import domainData

from .support_common import volumeToSupport, volumeToSupport_tensor, n_h_to_nH, nH_to_n_h



# Module-level Python floats: a `scalar_t(np.pi * 3.0 / 4.0)` (or
# `scalar_t(1.0/3.0)`) binop inside a traced func is evaluated in float32 by
# the warp tracer, silently corrupting the sphere-volume factor and the
# cube-root exponent at ~1e-8 relative in float64 builds (same gotcha as the
# B7 knots / the properties.py constant-BinOp finding, PLAN log 2026-09-18).
# Plain module constants referenced by name survive float64 exactly
# (probe-verified, .tmp/support_const_probe.py).
_VOL_3D_FACTOR = np.pi * 3.0 / 4.0   # (4/3)pi h^3 = N_H v  ->  h^3 = N_H v / (3pi/4)
_CUBE_ROOT_EXP = 1.0 / 3.0


@wp.func
def volumeToSupport_warp(volume : scalar_t, targetNeighbors : wp.int32, dim : wp.int32):
    # the tracer has no int32 x float64 mul overload; cast first (exact for
    # the integer neighbor counts involved)
    n = scalar_t(targetNeighbors)
    if dim == 1:
        return n * volume / scalar_t(2.0)
    elif dim == 2:
        return safe_sqrt(n * volume / scalar_t(np.pi))
    else:
        return wp.pow(n * volume / scalar_t(_VOL_3D_FACTOR), scalar_t(_CUBE_ROOT_EXP))



@wp.func
def computePairwiseSupport(hx: scalar_t, hy: scalar_t, mode: wp.uint32):
    if mode == wp.static(SupportScheme.Gather.value): # gather
        return hx
    elif mode == wp.static(SupportScheme.Scatter.value): # scatter
        return hy
    elif mode == wp.static(SupportScheme.MeanSymmetric.value): # meanSymmetric
        return (hx + hy) / scalar_t(2.0)
    else:
        return wp.max(hx, hy)


@wp.func
def computePairwiseSupportJVP(hx: scalar_t, hy: scalar_t, dhx: scalar_t, dhy: scalar_t, mode: wp.uint32):
    """JVP of `computePairwiseSupport` -- ordinary calculus, not kernel math
    (`warpier_adjoint.md` Tier 2.1). The `else` branch (`max(hx, hy)`) is a
    genuine subgradient, discontinuous at `hx == hy`; exact away from that
    kink, same class of forward-branch-boundary case as `pinv`'s rank
    cutoff (Tier 2.4)."""
    if mode == wp.static(SupportScheme.Gather.value):
        return dhx
    elif mode == wp.static(SupportScheme.Scatter.value):
        return dhy
    elif mode == wp.static(SupportScheme.MeanSymmetric.value):
        return (dhx + dhy) / scalar_t(2.0)
    else:
        if hx >= hy:
            return dhx
        return dhy


@wp.func
def isInSupport(
    xi: vector(dtype=scalar_t, length=3), xj: vector(dtype=scalar_t, length=3),
    hi: scalar_t, hj: scalar_t, mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool), minDomain: wp.array(dtype = scalar_t), maxDomain: wp.array(dtype = scalar_t)
    ):
    hij = computePairwiseSupport(hi, hj, mode)
    domainState = domainData()
    domainState.domainMin = minDomain
    domainState.domainMax = maxDomain
    domainState.periodicity = periodic
    domainState.dim = wp.int32(minDomain.shape[0])
    xij = computeDistanceVec(xi, xj, domainState)
    r = vectorNorm_warp(xij)
    q = r / hij
    return q <= scalar_t(1.0)