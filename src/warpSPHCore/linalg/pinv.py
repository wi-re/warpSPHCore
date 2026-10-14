"""Batched pseudo-inverse of symmetric 1x1 .. 3x3 matrices, sync-free.

The matrices SPH builds per particle (the gradient-correction moment `M = sum_j V_j x_ji (x) grad W`, covariance and renormalisation
matrices) are symmetric to round-off for a radial kernel; `torch.linalg.pinv` of them costs a host sync (it checks cuSOLVER's info flag)
and many small launches. Here one kernel per batch:

* a cyclic Jacobi eigen-decomposition of the matrix zero-padded to 3x3. The padding adds eigenvalues 0 that the cut-off drops, so one
  function serves every dimension (no per-dimension variants); 8 sweeps over the three off-diagonal pairs converge a 3x3 to round-off;
* the cut-off is `torch.linalg.pinv`'s: eigenvalues with `|lambda| <= rtol * max|lambda|` are dropped, `rtol = d * eps(dtype)` by default
  (pass `rtol` for a different policy, e.g. `1e-6` for the renormalisation matrices);
* eigenvalues are signed (`1 / lambda`, not `1 / |lambda|`), so indefinite and negative-definite matrices invert like torch's SVD pinv does.

Only the symmetric part is used (`(S + S^T) / 2`): a clearly non-symmetric matrix is NOT inverted correctly -- use `torch.linalg.pinv` for those.
Accuracy against a float64 pinv is the same as torch's own float32 pinv (rows with condition number below 1e3: 1.7e-5 / 3.1e-5 relative in
2D / 3D float32, ~1e-13 in float64); rows that are ill-conditioned to within the cut-off (cond >~ 2e4 in float32) are noise in both and can
differ. Forward only: with an input that requires grad the torch call is used instead. Not `warpSPHCore.pinv` (the renormalisation wrapper), which is a
renormalisation policy (identity below four neighbours, rcond 1e-6) with no 3x3.

Inside a kernel: `A = loadPadded3x3(arr2d, i, dim)`, `P = symmetricPinv3(A, rtol)`.
"""

from typing import Optional

import torch
import warp as wp
from warp.types import matrix

from ..type_config import scalar_t

__all__ = ['pinvSymmetric', 'applyPinv', 'symmetricPinv3', 'loadPadded3x3']

_JACOBI_SWEEPS = 8


@wp.func
def _rotate(A: matrix(shape=(3, 3), dtype=scalar_t), V: matrix(shape=(3, 3), dtype=scalar_t), p: int, q: int):
    """One Jacobi rotation that zeroes A[p, q]; returns the rotated A and the accumulated eigenvector matrix."""
    apq = A[p, q]
    if wp.abs(apq) > scalar_t(0.0):
        theta = (A[q, q] - A[p, p]) / (scalar_t(2.0) * apq)
        sgn = scalar_t(1.0)
        if theta < scalar_t(0.0):
            sgn = scalar_t(-1.0)
        t = sgn / (wp.abs(theta) + wp.sqrt(theta * theta + scalar_t(1.0)))
        c = scalar_t(1.0) / wp.sqrt(t * t + scalar_t(1.0))
        s = t * c
        R = wp.identity(n=3, dtype=scalar_t)
        R[p, p] = c
        R[q, q] = c
        R[p, q] = s
        R[q, p] = -s
        A = wp.transpose(R) * A * R
        V = V * R
    return A, V


@wp.func
def symmetricPinv3(S: matrix(shape=(3, 3), dtype=scalar_t), rtol: scalar_t):
    """Pseudo-inverse of the symmetric part of the (zero-padded) 3x3 matrix `S`; see the module doc."""
    A = scalar_t(0.5) * (S + wp.transpose(S))
    V = wp.identity(n=3, dtype=scalar_t)
    for sweep in range(_JACOBI_SWEEPS):
        A, V = _rotate(A, V, 0, 1)
        A, V = _rotate(A, V, 0, 2)
        A, V = _rotate(A, V, 1, 2)
    big = wp.max(wp.abs(A[0, 0]), wp.max(wp.abs(A[1, 1]), wp.abs(A[2, 2])))
    inv = wp.matrix(shape=(3, 3), dtype=scalar_t)
    if big > scalar_t(0.0):
        cut = rtol * big
        for k in range(3):
            lam = A[k, k]
            if wp.abs(lam) <= cut:
                continue
            for a in range(3):
                for b in range(3):
                    inv[a, b] = inv[a, b] + V[a, k] * V[b, k] / lam
    return inv


@wp.func
def loadPadded3x3(M: wp.array2d(dtype=scalar_t), i: int, dim: int):
    """Row `i` of a `(n, dim * dim)` array as a 3x3 matrix, zero-padded beyond `dim`."""
    out = wp.matrix(shape=(3, 3), dtype=scalar_t)
    for a in range(3):
        for b in range(3):
            if a < dim and b < dim:
                out[a, b] = M[i, a * dim + b]
    return out


@wp.kernel
def _pinvKernel(M: wp.array2d(dtype=scalar_t), dim: int, rtol: scalar_t, out: wp.array2d(dtype=scalar_t)):
    i = wp.tid()
    P = symmetricPinv3(loadPadded3x3(M, i, dim), rtol)
    for a in range(3):
        for b in range(3):
            if a < dim and b < dim:
                out[i, a * dim + b] = P[a, b]


@wp.kernel
def _applyPinvKernel(V: wp.array3d(dtype=scalar_t), M: wp.array2d(dtype=scalar_t), dim: int, rtol: scalar_t,
                     out: wp.array3d(dtype=scalar_t)):
    """out[i] = V[i] @ pinv(M[i]) for `(n, p, dim)` V (any number of rows p)."""
    i = wp.tid()
    P = symmetricPinv3(loadPadded3x3(M, i, dim), rtol)
    for a in range(V.shape[1]):
        for c in range(3):
            if c < dim:
                acc = scalar_t(0.0)
                for b in range(3):
                    if b < dim:
                        acc = acc + V[i, a, b] * P[b, c]
                out[i, a, c] = acc


def _check(M: torch.Tensor):
    if M.ndim != 3 or M.shape[1] != M.shape[2] or not 1 <= M.shape[1] <= 3:
        raise ValueError(f'expected a batch of symmetric 1x1..3x3 matrices, shape (n, d, d) with d <= 3, got {tuple(M.shape)}')


def _rtol(rtol: Optional[float], dim: int, dtype: torch.dtype) -> float:
    return dim * torch.finfo(dtype).eps if rtol is None else rtol         # torch.linalg.pinv's default: max(m, n) * eps


def pinvSymmetric(M: torch.Tensor, rtol: Optional[float] = None) -> torch.Tensor:
    """`torch.linalg.pinv` of a batch `(n, d, d)` of symmetric matrices (d <= 3): one launch, no host read."""
    _check(M)
    if M.requires_grad or M.shape[0] == 0:
        return torch.linalg.pinv(M, rtol=rtol)
    n, dim, _ = M.shape
    out = torch.empty_like(M)
    wp.launch(_pinvKernel, dim=n, device=wp.device_from_torch(M.device),
              inputs=[wp.from_torch(M.reshape(n, dim * dim).contiguous()), dim, _rtol(rtol, dim, M.dtype),
                      wp.from_torch(out.view(n, dim * dim))])
    return out


def applyPinv(V: torch.Tensor, M: torch.Tensor, rtol: Optional[float] = None) -> torch.Tensor:
    """`V @ pinv(M)` per batch entry, `V` of shape `(n, p, d)` and `M` of shape `(n, d, d)` symmetric: the
    `einsum('ipj,ijk->ipk', V, torch.linalg.pinv(M))` of a gradient corrected by its moment matrix, in one launch with no host read."""
    _check(M)
    if V.ndim != 3 or V.shape[0] != M.shape[0] or V.shape[2] != M.shape[1]:
        raise ValueError(f'V must be (n, p, d) for M (n, d, d); got {tuple(V.shape)} and {tuple(M.shape)}')
    if V.requires_grad or M.requires_grad or M.shape[0] == 0:
        return torch.einsum('ipj,ijk->ipk', V, torch.linalg.pinv(M, rtol=rtol))
    n, dim, _ = M.shape
    out = torch.empty_like(V)
    wp.launch(_applyPinvKernel, dim=n, device=wp.device_from_torch(M.device),
              inputs=[wp.from_torch(V.contiguous()), wp.from_torch(M.reshape(n, dim * dim).contiguous()), dim,
                      _rtol(rtol, dim, M.dtype), wp.from_torch(out)])
    return out
