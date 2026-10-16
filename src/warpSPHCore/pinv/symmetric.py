"""Fixed-branch derivatives of a symmetric selected spectral inverse."""

import torch


def _symmetric_pinv_jvp(C: torch.Tensor, L: torch.Tensor, dC: torch.Tensor) -> torch.Tensor:
    """Differentiate the selected spectral inverse, with its retained set fixed.

    Plain radial-kernel covariance is symmetric; the 2D primal explicitly
    symmetrizes it. For retained eigenvalue a and discarded eigenvalue b,
    the cross-eigenspace coefficient is 1 / (a * (a - b)), not zero and
    not 1 / a**2 unless b is exactly zero. Avoid eigenvectors here so repeated
    retained eigenvalues (including isotropic full-rank neighborhoods) stay
    smooth. In dimensions <= 3 a partial retained set has either one retained
    or one discarded direction, allowing small nonsingular projected solves.
    Cutoff crossings themselves have no derivative.
    """
    C = (C + C.mT) * .5
    dC = (dC + dC.mT) * .5
    result = -L @ dC @ L
    dim = C.shape[-1]
    if dim == 1:
        return result
    identity = torch.eye(dim, dtype=C.dtype, device=C.device)
    P = C @ L
    P = (P + P.mT) * .5
    Q = identity - P
    rank = P.diagonal(dim1=-2, dim2=-1).sum(-1).round()
    # Torch's small solve has no half-precision backend. Preserve the public
    # tangent dtype while doing only this local solve in a supported dtype.
    solve_dtype = torch.float32 if C.dtype in (torch.float16, torch.bfloat16) else C.dtype

    def solve(matrix, rhs):
        # Rank selection separates retained/discarded eigenvalues, so these
        # projected systems are nonsingular. Avoid solve's CUDA error readback
        # to retain current-stream/capture behavior for explicit traversals.
        return torch.linalg.solve_ex(
            matrix.to(solve_dtype), rhs.to(solve_dtype), check_errors=False,
        ).result.to(C.dtype)

    one_retained = (rank == 1).view(-1, 1, 1)
    a = (C @ P).diagonal(dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
    denominator = torch.where(one_retained, a * Q - Q @ C @ Q + P, identity)
    cross = solve(denominator, torch.where(one_retained, Q @ dC @ L, 0.))
    if dim == 3:
        one_discarded = (rank == 2).view(-1, 1, 1)
        b = (C @ Q).diagonal(dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
        denominator = torch.where(one_discarded, P @ C @ P - b * P + Q, identity)
        cross = cross + solve(
            denominator, torch.where(one_discarded, L @ dC @ Q, 0.))
    return result + cross + cross.mT


class _SelectedSymmetricInverse(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(matrix, rtol):
        return torch.linalg.pinv((matrix + matrix.mT) * .5, rtol=rtol, hermitian=True)

    @staticmethod
    def setup_context(ctx, inputs, output):
        matrix, _ = inputs
        # Save the original input and output: their graph links are needed by
        # reverse-over-JVP and higher reverse derivatives of the fixed branch.
        ctx.save_for_backward(matrix, output)
        ctx.save_for_forward(matrix, output)

    @staticmethod
    def backward(ctx, cotangent):
        matrix, inverse = ctx.saved_tensors
        # On symmetric matrices the spectral Frechet derivative is self-adjoint.
        return _symmetric_pinv_jvp(matrix, inverse, cotangent), None

    @staticmethod
    def jvp(ctx, tangent, tangent_rtol):
        matrix, inverse = ctx.saved_tensors
        return _symmetric_pinv_jvp(matrix, inverse, tangent)


def selectedSymmetricInverse(matrix, rtol=1e-6):
    """Symmetric spectral inverse with a fixed retained-set AD rule.

    The partial-rank rule includes nonzero discarded eigenvalues. Rank/cutoff
    switches remain nonsmooth. This internal correction helper supports d<=3;
    it does not replace the general nonsymmetric pseudo-inverse API.
    """
    return _SelectedSymmetricInverse.apply(matrix, rtol)
