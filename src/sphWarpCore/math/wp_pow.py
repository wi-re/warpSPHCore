import warp as wp
from ..type_config import scalar_t
import numpy as np
    
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