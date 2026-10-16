"""Owned diagonal/block inverse actions with explicit setup and failure policy.

Setup/update may synchronize. Application uses only Torch device operations;
fixed-address coefficient updates are visible to already-captured applications.
"""

from typing import Any

import torch
import warp as wp
from torch.autograd import forward_ad

from ..autograd.stream import currentTorchWarpStream
from .krylov.common import finiteScalar
from .operator import LinearOperator

__all__ = ['DiagonalPreconditioner', 'BlockJacobiPreconditioner']


@wp.kernel(enable_backward=False)
def _blockApply(inverse: wp.array3d(dtype=Any), v: wp.array(dtype=Any),
                out: wp.array(dtype=Any), d: int, transpose: bool):
    row = wp.tid()
    particle, component = row // d, row % d
    value = v.dtype(0)
    for k in range(d):
        coefficient = inverse[particle, component, k]
        if transpose:
            coefficient = inverse[particle, k, component]
        value += coefficient * v[particle * d + k]
    out[row] = value


def _check(values, block):
    if not isinstance(values, torch.Tensor):
        raise ValueError('coefficients must be a tensor')
    if block:
        if values.ndim != 3 or values.shape[1] != values.shape[2] or values.shape[1] < 1:
            raise ValueError('blocks must have shape (n, d, d), with d >= 1')
    elif values.ndim != 1:
        raise ValueError('diagonal must be a flat tensor')
    if values.dtype not in (torch.float32, torch.float64) or values.device.type not in ('cpu', 'cuda'):
        raise ValueError('coefficients must be CPU/CUDA float32 or float64')
    if values.requires_grad or forward_ad.unpack_dual(values).tangent is not None:
        raise ValueError('preconditioner setup/update does not differentiate coefficients')
    if values.device.type == 'cuda':
        with torch.cuda.device(values.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('preconditioner setup/update must run outside CUDA capture')


def _reciprocals(values, cutoff):
    keep = values.abs() > cutoff
    reciprocal = torch.reciprocal(torch.where(keep, values, torch.ones_like(values)))
    keep = keep & torch.isfinite(reciprocal)
    return torch.where(keep, reciprocal, torch.zeros_like(reciprocal)), keep


class _Preconditioner:
    _block = False

    def __init__(self, coefficients, *, atol=0.0, rtol=None, shift=0.0,
                 singular='error', requireSPD=False, application='torch'):
        _check(coefficients, self._block)
        for name, value in (('atol', atol), ('shift', shift)):
            finiteScalar(name, value, nonnegative=name != 'shift')
        if rtol is not None:
            finiteScalar('rtol', rtol)
        if singular not in ('error', 'identity', 'pinv'):
            raise ValueError("singular must be 'error', 'identity' or 'pinv'")
        if not isinstance(requireSPD, bool):
            raise ValueError('requireSPD must be bool')
        self._blockSize = coefficients.shape[1] if self._block else 1
        if application not in ('torch', 'native'):
            raise ValueError("application must be 'torch' or 'native'")
        if application == 'native' and (not self._block or self._blockSize > 3):
            raise ValueError('native application supports only 1x1..3x3 blocks')
        self._application = application
        self._coefficientShape = coefficients.shape
        self._atol = float(atol)
        self._rtol = self._blockSize * torch.finfo(coefficients.dtype).eps if rtol is None else float(rtol)
        self._shift = float(shift)
        self._singular = singular
        self._requireSPD = requireSPD
        inverse, count = self._build(coefficients)
        self._inverse = inverse
        self._fallbackCount = count
        self._generation = 0
        n = coefficients.shape[0] * self._blockSize
        self._operator = LinearOperator((n, n), inverse.dtype, inverse.device,
                                        self._apply, self._applyOut, self._applyTranspose)
        self._transpose = LinearOperator((n, n), inverse.dtype, inverse.device,
                                         self._applyTranspose, self._applyTransposeOut, self._apply)

    def _build(self, coefficients):
        # No input storage is retained. Construct and validate the entire new
        # action before mutating any live coefficient buffer in update().
        shifted = coefficients.clone()
        if self._block:
            shifted = shifted + self._shift * torch.eye(self._blockSize, device=shifted.device, dtype=shifted.dtype)
        else:
            shifted = shifted + self._shift
        if not bool(torch.isfinite(shifted).all()):
            raise ValueError('shifted preconditioner coefficients must be finite')
        if not self._block:
            cutoff = torch.maximum(shifted.abs() * self._rtol, shifted.new_tensor(self._atol))
            inverse, keep = _reciprocals(shifted, cutoff)
            bad = ~keep
        elif shifted.shape[0] == 0:
            return shifted.contiguous(), 0
        else:
            if self._requireSPD:
                # Eigh is valid only under this explicit, exact symmetry
                # assertion. Nonsymmetric blocks always use general SVD.
                if not torch.equal(shifted, shifted.transpose(-1, -2)):
                    raise ValueError('requireSPD needs exactly symmetric blocks; symmetrize explicitly if intended')
                spectrum, vectors = torch.linalg.eigh(shifted)
                left, right = vectors, vectors.transpose(-1, -2)
            else:
                U, spectrum, Vh = torch.linalg.svd(shifted, full_matrices=False)
                left, right = Vh.transpose(-1, -2), U.transpose(-1, -2)
            cutoff = torch.maximum(self._rtol * spectrum.abs().amax(dim=-1, keepdim=True),
                                   spectrum.new_tensor(self._atol))
            reciprocal, keep = _reciprocals(spectrum, cutoff)
            bad = ~keep.all(dim=-1)
            inverse = (left * reciprocal.unsqueeze(-2)) @ right
            if self._requireSPD:
                # Preserve a symmetric inverse despite GEMM rounding, so the
                # certified action is suitable for CG's SPD contract.
                inverse = .5 * inverse + .5 * inverse.transpose(-1, -2)
        count = int(bad.sum())
        if count and self._singular == 'error':
            raise ValueError(f'{count} unusable diagonal entries/blocks; choose an explicit singular policy or shift')
        if count and self._singular == 'identity':
            identity = (torch.eye(self._blockSize, dtype=inverse.dtype, device=inverse.device)
                        if self._block else torch.ones_like(inverse))
            mask = bad[:, None, None] if self._block else bad
            inverse = torch.where(mask, identity, inverse)
        if not bool(torch.isfinite(inverse).all()):
            raise ValueError('preconditioner inverse is not finite in the coefficient dtype')
        if self._requireSPD:
            if self._block:
                # A truncated inverse is semidefinite even if rounding happens
                # to make a numerical Cholesky appear successful.
                if count and self._singular == 'pinv':
                    raise ValueError('requireSPD forbids rank-deficient pseudoinverse actions')
                _, info = torch.linalg.cholesky_ex(inverse, check_errors=False)
                positive = bool((info == 0).all())
            else:
                positive = bool((inverse > 0).all())
            if not positive:
                raise ValueError('requireSPD needs a positive definite inverse action')
        return inverse.contiguous(), count

    @property
    def shape(self):
        return self._operator.shape

    @property
    def dtype(self):
        return self._operator.dtype

    @property
    def device(self):
        return self._operator.device

    @property
    def inverse(self):
        """Owned snapshot for diagnostics; mutating it cannot change this action."""
        return self._inverse.clone()

    @property
    def nbytes(self):
        return self._inverse.numel() * self._inverse.element_size()

    @property
    def fallbackCount(self):
        """Number of unusable entries/blocks in the last successful setup."""
        return self._fallbackCount

    @property
    def generation(self):
        """Number of successful updates since construction."""
        return self._generation

    @property
    def spdVerified(self):
        """True only when requireSPD explicitly certified the current action."""
        return self._requireSPD

    def asLinearOperator(self):
        """Stable callable adapter; updates to this helper remain visible."""
        return self._operator

    def update(self, coefficients):
        """Validate then replace coefficients in-place, retaining captured addresses.

        Shape/dtype/device and all construction policies remain fixed. A
        failed validation leaves the previous action and diagnostics intact.
        Do not update during a solve or concurrently with an application.
        """
        _check(coefficients, self._block)
        if (coefficients.shape != self._coefficientShape or coefficients.dtype != self.dtype
                or coefficients.device != self.device):
            raise ValueError('update must preserve coefficient shape, dtype and device')
        inverse, count = self._build(coefficients)
        self._inverse.copy_(inverse)
        self._fallbackCount = count
        self._generation += 1
        return self

    def _multiply(self, v, out=None, *, transpose=False):
        if not self._block:
            return torch.mul(self._inverse, v, out=out)
        ad = v.requires_grad or forward_ad.unpack_dual(v).tangent is not None
        outAD = out is not None and (out.requires_grad or forward_ad.unpack_dual(out).tangent is not None)
        if self._application == 'native' and v.device.type == 'cuda' and not ad and not outAD:
            result = torch.empty_like(v) if out is None else out
            if v.numel():
                wp.launch(_blockApply, dim=v.numel(),
                          inputs=[wp.from_torch(self._inverse), wp.from_torch(v),
                                  wp.from_torch(result), self._blockSize, transpose],
                          stream=currentTorchWarpStream(v.device))
            return result
        matrix = self._inverse.transpose(-1, -2) if transpose else self._inverse
        vectors = v.reshape(-1, self._blockSize, 1)
        if out is None:
            return torch.bmm(matrix, vectors).reshape(-1)
        torch.bmm(matrix, vectors, out=out.view(-1, self._blockSize, 1))
        return out

    def _apply(self, v):
        return self._multiply(v)

    def _applyOut(self, v, out):
        return self._multiply(v, out)

    def _applyTranspose(self, v):
        return self._multiply(v, transpose=True)

    def _applyTransposeOut(self, v, out):
        return self._multiply(v, out, transpose=True)

    def __call__(self, v):
        return self._operator(v)

    def apply(self, v, out=None):
        return self._operator.apply(v, out)

    def applyAdjoint(self, v, out=None):
        return self._transpose.apply(v, out)


class DiagonalPreconditioner(_Preconditioner):
    """Inverse of a supplied operator diagonal (not an already-inverted vector).

    Cutoff per entry is max(atol, rtol * abs(diagonal + shift)); entries are
    independent 1x1 blocks, so unrelated row scales do not discard each other.
    Default rtol is eps(dtype). Unusable reciprocals obey singular='error',
    'identity' (inverse one) or 'pinv' (inverse zero). No ridge is implicit.
    """


class BlockJacobiPreconditioner(_Preconditioner):
    """Inverse action of (n, d, d) diagonal blocks on particle-major flat vectors.

    Setup uses general SVD, or signed symmetric eigenvalues under requireSPD.
    Per-block cutoff is max(atol, rtol * largest singular/eigenvalue magnitude),
    with default rtol=d*eps(dtype). 'identity' replaces an entire unusable
    block; 'pinv' truncates only discarded components. requireSPD checks the
    resulting action, allowing identity fallback but rejecting truncated rank.
    application='native' opts into a dtype-generic CUDA kernel for d <= 3;
    CPU and vector-AD calls retain Torch. Warm it before CUDA graph capture.
    """

    _block = True
