import warp as wp
import numpy as np

@wp.func
def computePairwiseSupport(hx: wp.float32, hy: wp.float32, mode: wp.uint32):
    if mode == 0: # gather
        return hx
    elif mode == 1: # scatter
        return hy
    elif mode == 2: # symmetric
        return (hx + hy) / 2.0
    elif mode == 3: # superSymmetric
        return wp.max(hx, hy)
    else:
        return hx
    
    
@wp.func
def iPow(base: wp.float32, exp: int):
    if exp == 0:
        return 1.0
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
        return base**(wp.float32(exp))
    return np.nan



@wp.func
def cpow_warp(q: wp.float32, p: int):
    return iPow(wp.clamp(q, 0.0, 1.0),p)


from typing import Any
@wp.func
def get_epsilon(val: Any):
    return 1e-7

# @wp.overload
# def get_epsilon(val: wp.float32):
#     return 1e-7

# @wp.overload
# def get_epsilon(val: wp.float64):
#     return 1e-15