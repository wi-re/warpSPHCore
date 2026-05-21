import warp as wp
import numpy as np

from ..enumTypes import *
from ..type_config import scalar_t
from ..types import *

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
def iPow(base: scalar_t, exp: int):
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
        return base ** (scalar_t(exp))
    return scalar_t(np.nan)



@wp.func
def cpow_warp(q: scalar_t, p: int):
    return iPow(wp.clamp(q, scalar_t(0.0), scalar_t(1.0)), p)

@wp.func
def bpow_warp(q: scalar_t, p: int):
    return iPow(wp.min(q, scalar_t(0.0)), p)


from typing import Any
@wp.func
def get_epsilon(val: Any):
    return scalar_t(1e-7)

@wp.func
def get_epsilon(val: wp.float32) -> wp.float32:
    return wp.float32(1e-7)

@wp.func
def get_epsilon(val: wp.float64) -> wp.float64:
    return wp.float64(1e-15)

@wp.func
def get_epsilon(val: wp.float16) -> wp.float16:
    return wp.float16(1e-3)