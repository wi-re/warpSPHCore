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
from ..profiling import record_function

from ..type_config import scalar_t
from ..util import *
from ..math import *
from ..autograd import warpWrapper, launch_kernel

# wp.mat22f/wp.vec2f (etc.) are warp's own built-in named 2x2-matrix/length-2-vector
# types, one per precision -- unlike the 1x1 case (see wp_pinv1x1.py's comment), warp
# already ships named subclasses for 2x2/vec2 so there's no need for warpSPHCore to
# define its own; picked here by scalar_t for the same reason wp_pinv1x1.py picks
# mat11f/vec1f -- launch_kernel's wp.zeros(..., dtype=...) needs a concrete, hashable
# type, and it must match scalar_t rather than being hardcoded to float32.
if scalar_t == wp.float32:
    _mat22_t, _vec2_t = wp.mat22f, wp.vec2f
elif scalar_t == wp.float64:
    _mat22_t, _vec2_t = wp.mat22d, wp.vec2d
elif scalar_t == wp.float16:
    _mat22_t, _vec2_t = wp.mat22h, wp.vec2h
else:
    raise ValueError(f"Unsupported scalar type: {scalar_t}")


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
    # Inputs first, outputs last -- launch_kernel builds its kernel_inputs list as
    # inputs + outputs, so the parameter order here must match that convention (see
    # wp_pinv1x1.py's pseudoInverse1x1Kernel for the same shape).
    C: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)), # type: ignore
    num_nbrs: wp.array(dtype=wp.int32),  # type: ignore
    L: wp.array(dtype=matrix(shape=(2,2), dtype=scalar_t)), # type: ignore
    EV: wp.array(dtype=vector(length=2, dtype=scalar_t)), # type: ignore
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

    # Eigenvalue MAGNITUDES via the closed-form trace/determinant quadratic (atan2-free) --
    # used only to decide, below, whether C is well-conditioned (the common case: a regular/
    # well-resolved particle neighborhood) or needs an eigenvalue truncated (rank-deficient,
    # e.g. too few effectively-independent neighbor directions). This decision is forward-only
    # (no gradient needs to flow through an `if`, same as every other branch condition in this
    # codebase), so `disc`'s own kink at disc==0 (a==d and b==0 simultaneously) is harmless here.
    trace = a + d
    diff = a - d
    # safe_sqrt (not wp.sqrt): at exact isotropy (diff==0, b==0) the radicand is exactly 0,
    # where sqrt's own derivative is infinite -- disc's value isn't consumed by the
    # well-conditioned branch's L computation below at all (only by the branch condition and
    # the EV diagnostic output), but Warp's autodiff still computes and accumulates its adjoint
    # contribution regardless of whether the caller reads EV, and an infinite adjoint here
    # corrupts the shared upstream a/b/d adjoint accumulators with NaN even though L's own
    # contribution is perfectly finite (confirmed empirically: without safe_sqrt,
    # gradcheck's analytical Jacobian was all-NaN at C==I even though L is smooth there).
    # safe_sqrt's own custom `@wp.func_grad` (math/wp_sqrt.py) already handles exactly this
    # class of "true value needed, but adjoint undefined at x==0" case elsewhere in this
    # codebase by contributing a zero adjoint there instead.
    disc = safe_sqrt(diff * diff + scalar_t(4.0) * b * b)
    lamP = scalar_t(0.5) * (trace + disc)
    lamM = scalar_t(0.5) * (trace - disc)
    big = lamP
    small = lamM
    if wp.abs(lamM) > wp.abs(lamP):
        big = lamM
        small = lamP

    EV[i][0] = big
    EV[i][1] = small

    # Zeroing based on a fixed absolute epsilon lets thin/anisotropic neighborhoods (e.g. free-surface
    # fingers, near-collinear particle rows) through with a tiny but nonzero small eigenvalue, which
    # then gets inverted into a huge amplification factor. Use a cutoff relative to the largest
    # eigenvalue instead, matching the rcond convention torch.linalg.pinv uses for the 3D path.
    rcond = scalar_t(1.0e-6)
    threshold = rcond * wp.abs(big)

    if wp.abs(big) > scalar_t(1.0e-12) and wp.abs(small) > threshold:
        # Well-conditioned (full-rank) C: pinv(C) is mathematically IDENTICAL to the ordinary
        # matrix inverse here (a pseudo-inverse only differs from the ordinary inverse when a
        # singular value gets truncated, which this branch's own condition just ruled out) --
        # so compute it via the direct closed-form 2x2 inverse instead of reconstructing from
        # eigenvectors. This formula is smooth in (a,b,d) everywhere det != 0, with no atan2
        # anywhere, fixing a genuine reverse-mode adjoint bug the eigenvector-reconstruction
        # path below has at exactly isotropic C (a==d, b==0 -- the common case for a locally
        # regular/uniform particle neighborhood, e.g. any perfectly regular grid): Warp's own
        # `adj_atan2` silently drops the gradient contribution (treats it as exactly 0) when
        # both of `atan2`'s arguments are exactly 0, even though the TRUE derivative of pinv in
        # that direction is finite and well-defined -- confirmed via
        # `torch.autograd.gradcheck(pinv2x2_warpBackend, (C,))` at `C = I`: the numerical
        # Jacobian's `d(inv01)/dC01` is `-0.5`, not the `0` the eigenvector-reconstruction path
        # silently produced. This also removes the need for every renorm-touching gradcheck/
        # spike script's own `+-15%` non-uniform-support perturbation workaround for a perfectly
        # regular grid -- that workaround dodged this exact bug rather than fixing it.
        det = a * d - b * b
        invDet = scalar_t(1.0) / det
        L[i][0,0] = d * invDet
        L[i][0,1] = -b * invDet
        L[i][1,0] = -b * invDet
        L[i][1,1] = a * invDet
        return

    # Rank-deficient (or exactly-zero) C: fall back to the eigenvector reconstruction, needed
    # here (unlike the well-conditioned branch above) to actually zero out the near-null
    # eigendirection. This branch is rare in practice, and a rank-deficient C is inherently
    # anisotropic (one eigenvalue near zero, the other not) -- essentially never coinciding with
    # the exact isotropy (a==d, b==0) that makes the atan2-based eigenvector computation below
    # adjoint-fragile, so that fragility is not a practical concern in this branch. Closed-form
    # symmetric 2x2 eigendecomposition: a single atan2 call. This replaces a general (and for a
    # symmetric input, unnecessary) 2x2 SVD that computed U's rotation angle and V's rotation
    # angle from two SEPARATE atan2 expressions. For a near-isotropic C both of those
    # expressions' denominators round to ~0, and because the two expressions round differently
    # at the float-noise level, the two angles could land on unrelated values instead of the
    # (here) required theta==phi, producing an inverse that was spuriously rotated by tens of
    # degrees instead of staying diagonal -- reproduced directly against production covariance
    # matrices, see warpier_core.md. A symmetric matrix only has one rotation angle in the first
    # place, so computing it once removes the possibility of the two desyncing.
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
    bigEig = lam1
    smallX = v2x
    smallY = v2y
    smallEig = lam2
    if wp.abs(lam2) > wp.abs(lam1):
        bigX = v2x
        bigY = v2y
        bigEig = lam2
        smallX = v1x
        smallY = v1y
        smallEig = lam1

    big_inv = scalar_t(0.0) if wp.abs(bigEig) <= scalar_t(1.0e-12) else scalar_t(1.0) / bigEig
    small_inv = scalar_t(0.0) if wp.abs(smallEig) <= threshold else scalar_t(1.0) / smallEig

    L[i][0,0] = big_inv * bigX * bigX + small_inv * smallX * smallX
    L[i][0,1] = big_inv * bigX * bigY + small_inv * smallX * smallY
    L[i][1,0] = big_inv * bigY * bigX + small_inv * smallY * smallX
    L[i][1,1] = big_inv * bigY * bigY + small_inv * smallY * smallY

def pinv2x2_warpBackend(
    C: torch.Tensor,
    num_nbrs: torch.Tensor
):
    # Previously a raw wp.launch on cast tensors -- not wrapped in a torch.autograd.Function
    # at all, so it had no backward pass to gradcheck in the first place (found while closing
    # out the "pinv2x2_warpBackend has no gradcheck coverage" item in warpier_core.md). Ported
    # to the same warpWrapper/launch_kernel pattern wp_pinv1x1.py's pinv1x1 already used.
    outputSize = C.shape[0]
    inv, evs = warpWrapper(
        launch_kernel, pinv2x2_warp, outputSize, (_mat22_t, _vec2_t),
        C, num_nbrs
    )
    return inv, evs
