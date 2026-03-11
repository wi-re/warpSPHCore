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

@wp.func
def k(q: wp.float32, dim: int = 2):
    if dim == 1:
        return cpow_warp(1.0 - q, 5) * (1.0 + 5.0 * q + 8.0 * iPow(q, 2))
    else:
        return cpow_warp(1.0 - q, 6) * (1.0 + 6.0 * q + (35.0 / 3.0) * iPow(q, 2))
    
@wp.func
def C_d(dim: int):
    if dim == 1:
        return 3.0 / 2.0
    elif dim == 2:
        return 9.0 / np.pi
    else:
        return 495.0 / (32.0 * np.pi)

@wp.func
def wendland4_2d(
    r: wp.float32,
    h: wp.float32,
    dim: int = 2
):
    return k(r / h, dim) * C_d(dim) / iPow(h, dim)