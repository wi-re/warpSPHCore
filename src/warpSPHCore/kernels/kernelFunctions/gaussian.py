import numpy as np
from ...math import cpow_warp, iPow, bpow_warp
from ...type_config import *
from typing import Any
import warp as wp
from warp.types import vector

# Gaussian kernel, truncated at 16 sigma (Dehnen & Aly 2012, Fig. 2 /
# Table 2). Convention: W(r) = N(0, sigma^2) with h_paper = 2 sigma and
# H = 16 sigma -> kernelScale = H/h = 8.0, sigma = H/16:
#   f(q) = exp(-0.5 (16 q)^2) = exp(-128 q^2),  q = r/H,  f(0) = 1.
# C_d = (128/pi)^(nu/2) (exact for the untruncated Gaussian; the
# truncation error is e^-128 ~ 3e-56, negligible). D&A footnote 10: any
# truncation invalidates strict FT non-negativity -- the induced dip is
# at the e^-128 level, below the kernel audit's 1e-6 noise floor.
# sigma2/H2 = 1/256 in all dims (eq. 8 convention).

@wp.func
def Gaussian_k(q: scalar_t, dim: wp.int32 = 3):
    return wp.where(q <= scalar_t(1.0), wp.exp(-scalar_t(128.0) * q * q), scalar_t(0.0))

@wp.func
def Gaussian_dkdq(q: scalar_t, dim: wp.int32 = 3):
    e = wp.exp(-scalar_t(128.0) * q * q)
    return wp.where(q <= scalar_t(1.0), -scalar_t(256.0) * q * e, scalar_t(0.0))

@wp.func
def Gaussian_d2kdq2(q: scalar_t, dim: wp.int32 = 3):
    e = wp.exp(-scalar_t(128.0) * q * q)
    return wp.where(q <= scalar_t(1.0), (scalar_t(65536.0) * q * q - scalar_t(256.0)) * e, scalar_t(0.0))

@wp.func
def Gaussian_d3kdq3(q: scalar_t, dim: wp.int32 = 3):
    e = wp.exp(-scalar_t(128.0) * q * q)
    return wp.where(q <= scalar_t(1.0), (scalar_t(196608.0) * q - scalar_t(16777216.0) * q * q * q) * e, scalar_t(0.0))

@wp.func
def Gaussian_C_d(dim: wp.int32):
    base = scalar_t(128.0) / scalar_t(np.pi)
    if dim == 1: return wp.sqrt(base)
    elif dim == 2: return base
    else: return base * wp.sqrt(base)

@wp.func # H/h = 1/(2 sqrt(sigma2/H2)), sigma2/H2 = 1/256 in all dims (sigma = H/16)
def Gaussian_kernelScale(dim: wp.int32 = 3):
    return scalar_t(8.0)

@wp.func # D&A Table 2 Gaussian row: N_h = 10, packing 1.337 (N_H = 5120 at the 16-sigma truncation)
def Gaussian_packingRatio():
    return scalar_t(1.337)
