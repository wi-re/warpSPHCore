"""Pseudo-inverse of a symmetric 2x2 matrix -- its own Warp operation, not part of the
covariance kernel. Used by wp_renormalization.py to turn a covariance matrix (from
wp_covariance.py) into a gradient-renormalization matrix, but has no dependency on how
that covariance matrix was computed, so it lives in its own file.

``pinv2x2_warp``/``pinv2x2_warpBackend`` is the production path (called from
wp_renormalization.py). ``pinv2x2`` is a pure-PyTorch mirror of the same closed-form
symmetric eigendecomposition, kept fixed in step with the Warp kernel for comparison/
testing (see warp_renorm.ipynb) rather than used on any production path.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import record_function

from ..type_config import scalar_t
from ..util import *
from ..math import *


@torch.compile
def pinv2x2(M):
    # Symmetric closed-form eigendecomposition -- see pinv2x2_warp below (the actual production
    # path) for the derivation and why the general 2x2-SVD formula this used to use is unstable for
    # near-isotropic covariance matrices. This function is currently unused (computeRenormalizationMatrices_
    # calls pinv2x2_warpBackend instead) but kept fixed in step with it since it operates on the same
    # symmetric covariance matrices.
    with record_function('Pseudo Inverse 2x2'):
        a = M[:,0,0]
        b = 0.5 * (M[:,0,1] + M[:,1,0])
        d = M[:,1,1]

        theta = 0.5 * torch.atan2(2 * b, a - d)
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        v1 = torch.stack([cosTheta, sinTheta], -1)
        v2 = torch.stack([-sinTheta, cosTheta], -1)

        lam1 = a * cosTheta**2 + 2 * b * cosTheta * sinTheta + d * sinTheta**2
        lam2 = (a + d) - lam1

        swap = lam2.abs() > lam1.abs()
        big = torch.where(swap, lam2, lam1)
        small = torch.where(swap, lam1, lam2)
        bigV = torch.where(swap.unsqueeze(-1), v2, v1)
        smallV = torch.where(swap.unsqueeze(-1), v1, v2)

        eigVals = torch.stack([big, small], -1)

        rcond = 1e-6
        threshold = rcond * big.abs()
        big_inv = torch.where(big.abs() > 1e-12, 1 / big, torch.zeros_like(big))
        small_inv = torch.where(small.abs() > threshold, 1 / small, torch.zeros_like(small))

        inv = big_inv[:, None, None] * torch.einsum('ni,nj->nij', bigV, bigV) \
            + small_inv[:, None, None] * torch.einsum('ni,nj->nij', smallV, smallV)
        return inv, eigVals


@wp.func
def matmul2(
    A: matrix(shape=(2,2), dtype=scalar_t),
    B: matrix(shape=(2,2), dtype=scalar_t)
):
    out = zero_like_warp(A)
    out[0,0] = A[0,0] * B[0,0] + A[0,1] * B[1,0]
    out[0,1] = A[0,0] * B[0,1] + A[0,1] * B[1,1]
    out[1,0] = A[1,0] * B[0,0] + A[1,1] * B[1,0]
    out[1,1] = A[1,0] * B[0,1] + A[1,1] * B[1,1]
    return out

@wp.kernel
def pinv2x2_warp(
    C: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)), # type: ignore
    L: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)), # type: ignore
    EV: wp.array(dtype=vector(length=2, dtype=scalar_t)), # type: ignore
    num_nbrs: wp.array(dtype=wp.int32)  # type: ignore
):
    i = wp.tid()
    a = C[i][0,0]
    # C is a sum of V_j * x_ij (x) gradW_ij; for any isotropic kernel gradW_ij is parallel to x_ij,
    # so C is symmetric by construction (a sum of symmetric x_ij (x) x_ij terms). b/c below can still
    # differ at the floating-point-noise level depending on neighbor summation order -- symmetrize
    # rather than treat that noise as signal.
    b = scalar_t(0.5) * (C[i][0,1] + C[i][1,0])
    d = C[i][1,1]

    if num_nbrs[i] < 4:
        L[i][0,0] = scalar_t(1.0)
        L[i][0,1] = scalar_t(0.0)
        L[i][1,0] = scalar_t(0.0)
        L[i][1,1] = scalar_t(1.0)
        EV[i][0] = scalar_t(1.0)
        EV[i][1] = scalar_t(1.0)
        return

    # Closed-form symmetric 2x2 eigendecomposition: a single atan2 call. This replaces a general (and
    # for a symmetric input, unnecessary) 2x2 SVD that computed U's rotation angle and V's rotation
    # angle from two SEPARATE atan2 expressions. For a near-isotropic C (a~=d, b~=c~=0 -- the common
    # case for a locally regular/well-resolved particle neighborhood) both of those expressions'
    # denominators round to ~0, and because the two expressions round differently at the float-noise
    # level, the two angles could land on unrelated values instead of the (here) required theta==phi,
    # producing an inverse that was spuriously rotated by tens of degrees instead of staying diagonal
    # -- reproduced directly against production covariance matrices, see warpier_core.md. A symmetric
    # matrix only has one rotation angle in the first place, so computing it once removes the
    # possibility of the two desyncing.
    theta = (scalar_t(0.5)) * wp.atan2(scalar_t(2.0) * b, a - d)
    cosTheta = wp.cos(theta)
    sinTheta = wp.sin(theta)

    v1x = cosTheta
    v1y = sinTheta
    v2x = -sinTheta
    v2y = cosTheta

    lam1 = a * v1x * v1x + scalar_t(2.0) * b * v1x * v1y + d * v1y * v1y
    lam2 = (a + d) - lam1

    # order by magnitude, largest first, matching the "o1 >= o2" convention the rest of this function
    # (and its callers) assume. Eigenvalues are signed here, unlike the old singular-value convention.
    bigX = v1x
    bigY = v1y
    big = lam1
    smallX = v2x
    smallY = v2y
    small = lam2
    if wp.abs(lam2) > wp.abs(lam1):
        bigX = v2x
        bigY = v2y
        big = lam2
        smallX = v1x
        smallY = v1y
        small = lam1

    EV[i][0] = big
    EV[i][1] = small

    # Zeroing based on a fixed absolute epsilon lets thin/anisotropic neighborhoods (e.g. free-surface
    # fingers, near-collinear particle rows) through with a tiny but nonzero small eigenvalue, which
    # then gets inverted into a huge amplification factor. Use a cutoff relative to the largest
    # eigenvalue instead, matching the rcond convention torch.linalg.pinv uses for the 3D path.
    rcond = scalar_t(1.0e-6)
    threshold = rcond * wp.abs(big)
    big_inv = scalar_t(0.0) if wp.abs(big) <= scalar_t(1.0e-12) else scalar_t(1.0) / big
    small_inv = scalar_t(0.0) if wp.abs(small) <= threshold else scalar_t(1.0) / small

    L[i][0,0] = big_inv * bigX * bigX + small_inv * smallX * smallX
    L[i][0,1] = big_inv * bigX * bigY + small_inv * smallX * smallY
    L[i][1,0] = big_inv * bigY * bigX + small_inv * smallY * smallX
    L[i][1,1] = big_inv * bigY * bigY + small_inv * smallY * smallY

def pinv2x2_warpBackend(
    C: torch.Tensor,
    num_nbrs: torch.Tensor
):
    mat_warp = castTorchToWarpAsBuiltins(C)
    inv = torch.empty_like(C)
    evs = torch.empty((C.shape[0], 2), device = C.device, dtype = C.dtype)

    inv_warp = castTorchToWarpAsBuiltins(inv)
    evs_warp = castTorchToWarpAsBuiltins(evs)
    nnbrs = castTorchToWarpAsBuiltins(num_nbrs)
    wp.launch(kernel=pinv2x2_warp, dim=mat_warp.shape[0], inputs=[mat_warp, inv_warp, evs_warp, nnbrs], device=inv_warp.device)
    return inv, evs
