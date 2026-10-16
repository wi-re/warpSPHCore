"""Read a cubic local fit and nonlinear interface states on an open 2D cloud."""

import argparse
import json

import torch
import warp as wp

from warpSPHCore import DomainDescription, KernelFunctions, ParticleState, get_torch_precision
from warpSPHCore.polyfit import PolyFit, Reconstructor


def run(*, device='cpu'):
    wp.init()
    axis = torch.linspace(-.3, .3, 5, dtype=get_torch_precision(), device=device)
    positions = torch.cartesian_prod(axis, axis)
    positions += .002*torch.arange(positions.numel(), device=device, dtype=axis.dtype).reshape_as(positions).sin()
    h = axis.new_full((len(positions),), 1.4)
    p = ParticleState(positions, h, torch.full_like(h, .15**2),
                      torch.zeros(len(h), dtype=torch.int32, device=device), torch.ones_like(h))
    domain = DomainDescription(axis.new_full((2,), -2), axis.new_full((2,), 2),
                               torch.zeros(2, dtype=torch.bool, device=device), 2)
    x, y = positions.unbind(-1)
    field = 1+x**3+.3*x*y+y**2
    exact_gradient = torch.stack((3*x**2+.3*y, .3*x+2*y), -1)
    errors = {}
    tol = .006 if axis.dtype == torch.float32 else 2e-8
    for constant, name in ((True, 'mls'), (False, 'labfm')):
        fit = PolyFit.build(p, domain, KernelFunctions.Wendland2, 3, constant=constant)
        if bool(fit.deficient.any()):
            raise AssertionError('cubic example needs resolved stencils')
        gradient = fit.gradient(field)
        torch.testing.assert_close(gradient, exact_gradient, rtol=tol, atol=tol)
        torch.testing.assert_close(fit.laplacian(field), 6*x+2, rtol=tol*3, atol=tol*3)
        errors[name] = float((gradient-exact_gradient).abs().max())

    # Reuse the illustrated guide's resolved open 1D jump, away from boundaries.
    k = torch.arange(64, dtype=axis.dtype, device=device)
    x = (k+.5+.12*(7.3*k).sin())/64
    edges = torch.cat((x.new_zeros(1), .5*(x[:-1]+x[1:]), x.new_ones(1)))
    p = ParticleState(x[:, None], torch.full_like(x, 5.5/64), edges.diff(),
                      torch.zeros(64, dtype=torch.int32, device=device), torch.ones_like(x))
    domain = DomainDescription(x.new_tensor([-2.]), x.new_tensor([2.]),
                               torch.zeros(1, dtype=torch.bool, device=device), 1)
    jump = (x >= .5).to(x.dtype)
    i, j = torch.arange(63, device=device), torch.arange(1, 64, device=device)
    delta = p.positions[j]-p.positions[i]
    straddle = jump[i] != jump[j]
    diagnostics = {}
    for name in ('teno', 'weno'):
        # Omitted adjacency covers the candidates' larger radii automatically.
        rec = getattr(Reconstructor, name)(p, domain, KernelFunctions.Wendland2)
        left, right = rec.interfaceStates(jump, i, j, delta)
        if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
            raise AssertionError('nonfinite interface states')
        # Weight selection is not a positivity or monotonicity theorem.
        diagnostics[name] = dict(pairs=len(i), fallback_particles=int(rec.last_fallback.sum()),
                                 interface_min=float(torch.minimum(left.min(), right.min())),
                                 interface_max=float(torch.maximum(left.max(), right.max())),
                                 jump_left=float(left[straddle][0]), jump_right=float(right[straddle][0]))
        torch.testing.assert_close(left[straddle], torch.zeros_like(left[straddle]), rtol=0, atol=.02)
        torch.testing.assert_close(right[straddle], torch.ones_like(right[straddle]), rtol=0, atol=.02)
    return dict(device=str(axis.device), dtype=str(axis.dtype), gradient_max_errors=errors,
                reconstruction=diagnostics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    print(json.dumps(run(device=args.device), indent=2))
