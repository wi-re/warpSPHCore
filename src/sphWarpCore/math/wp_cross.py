import warp as wp
from ..type_config import *
from warp.types import vector, matrix
from typing import Any

# Unified Curl kernel: same design as the unified Gradient/Divergence kernels (see
# warpier_core.md's "Working Prototype -> Production" section). Curl shares Gradient's
# correction-path machinery (CRK, grad-h, volume, renormalization) and differs only in
# the neighbor-level contraction: `curlProduct` (Levi-Civita / cross-product contraction)
# in place of Gradient's `outerTensorProduct`.

@wp.func
def curlProduct(
    T: vector(dtype = scalar_t, length=Any),  # type: ignore
    V: vector(dtype = scalar_t, length=3), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    # This used to `return -R`, per a comment claiming the negation was needed
    # to match the right-hand rule. Confirmed via operation_matrix.py --dim 3
    # that this made every 3D curl output the exact negative of the true
    # (right-hand-rule) curl -- the 2D overload below has no such negation
    # and is dimensionally consistent with the standard convention, so the
    # 3D `-R` was the actual bug, not a deliberate convention choice. See
    # warpier_core.md.
    R = type(output)(scalar_t(0.0))
    dim = wp.int32(3) # hardcoded as this is the overload for 3D.
    for s in range(stride):
        # Flattened locations for T[0][s], T[1][s], T[2][s]
        k0 = wp.int32(s)
        k1 = wp.int32(s + stride)
        k2 = wp.int32(s + 2 * stride)
        # Levi-Civita / cross-product logic:
        R[0 * stride + s] = V[1] * T[k2] - V[2] * T[k1];
        R[1 * stride + s] = V[2] * T[k0] - V[0] * T[k2];
        R[2 * stride + s] = V[0] * T[k1] - V[1] * T[k0];
    return R

@wp.func
def curlProduct(
    T: vector(dtype = scalar_t, length=Any),  # type: ignore
    V: vector(dtype = scalar_t, length=2), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    R = type(output)(scalar_t(0.0))
    dim = wp.int32(2) # hardcoded as this is the overload for 2D.
    for s in range(stride+1): # loop to stride+1: in 2D the output has one less dimension than the input
        k0 = wp.int32(s)
        k1 = wp.int32(s + stride+1)
        # 2D cross-product logic: collapses the first dimension of T and the dimension of V
        R[s] = V[0] * T[k1] - V[1] * T[k0];
    return R

@wp.func
def curlProduct(
    T: vector(dtype = scalar_t, length=Any),  # type: ignore
    V: vector(dtype = scalar_t, length=1), # type: ignore
    output: vector(dtype = scalar_t, length=Any), # type: ignore
    stride: wp.int32,
    inputElements: wp.int32, outputElements: wp.int32
):
    R = type(output)(scalar_t(0.0))
    # in 1D the curl is identically zero
    R[0] = scalar_t(0.0)
    return R
