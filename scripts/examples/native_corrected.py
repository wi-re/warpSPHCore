"""Warp-free CRK interpolation and gradient, using supplied query coefficients.

Run WARPSPHCORE_BACKEND=native python scripts/examples/native_corrected.py
--device cuda (or cpu). This tiny Torch initialization solve is illustrative;
the native operators consume coefficients supplied by the caller.
"""
import argparse
import sys

import torch
import warpSPHCore as sph


def query_corrections(query_positions, reference_positions, reference_volumes, support):
    """Fit a linear reproducing kernel and differentiate its coefficients."""
    def solve_at(position):
        displacement = position - reference_positions
        q = displacement.norm(dim=-1) / support
        weights = (7 / torch.pi) / support**2 * (1 - q)**4 * (1 + 4 * q)
        basis = torch.cat((torch.ones_like(q[:, None]), displacement), dim=1)
        moments = torch.einsum('n,ni,nj->ij', reference_volumes * weights, basis, basis)
        rhs = position.new_tensor([1., 0., 0.])
        coefficients = torch.linalg.solve(moments, rhs)
        return torch.cat((coefficients[:1], coefficients[1:] / coefficients[0]))

    coefficients = torch.stack([solve_at(position) for position in query_positions])
    derivatives = torch.stack([torch.autograd.functional.jacobian(solve_at, position)
                               for position in query_positions])
    # The derivative index comes first in gradB. Explicit contiguous buffers
    # are part of the native borrowed-pointer contract.
    return sph.CRKState(coefficients[:, 0].contiguous(), coefficients[:, 1:].contiguous(),
        derivatives[:, 0].contiguous(), derivatives[:, 1:].transpose(1, 2).contiguous())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if sph.backend != 'native':
        raise RuntimeError('Set WARPSPHCORE_BACKEND=native before importing the package')
    axis = torch.tensor([-.4, 0., .4], dtype=torch.float64)
    reference_positions = torch.cartesian_prod(axis, axis)
    query_positions = axis.new_tensor([[-.1, .08], [.12, -.06]])
    reference_volumes = axis.new_full((len(reference_positions),), .16)
    support = 1.3  # Every source in this small example lies inside the support.
    crk = query_corrections(query_positions, reference_positions, reference_volumes, support)
    crk = sph.CRKState(**{name: getattr(crk, name).detach().to(args.device)
                          for name in ('A', 'B', 'gradA', 'gradB')})

    def particles(positions):
        positions = positions.to(args.device)
        return sph.ParticleState(positions, positions.new_full((len(positions),), support),
            masses=None, densities=None,
            kinds=torch.zeros(len(positions), device=args.device, dtype=torch.int32))
    query, reference = particles(query_positions), particles(reference_positions)
    domain = sph.DomainDescription(axis.new_full((2,), -1).to(args.device),
        axis.new_ones(2).to(args.device), torch.zeros(2, device=args.device, dtype=torch.bool), 2)
    query_values = 1 + 2 * query.positions[:, 0] - 3 * query.positions[:, 1]
    reference_values = 1 + 2 * reference.positions[:, 0] - 3 * reference.positions[:, 1]
    graph = sph.radiusSearch(query, domain, reference)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    options = dict(referenceParticles=reference, adjacency=graph,
                   referenceVolumes=reference_volumes.to(args.device), crkState=crk)
    interpolation = sph.warpOperation(query, properties, domain, query_values, reference_values, **options)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Gradient, gradientMode=sph.GradientScheme.Naive,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    gradient = sph.warpOperation(query, properties, domain, query_values, reference_values, **options)
    torch.testing.assert_close(interpolation, query_values, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(gradient, gradient.new_tensor([2., -3.]).expand_as(gradient), rtol=2e-12, atol=2e-12)
    assert not any(name == 'warp' or name.startswith('warp.') for name in sys.modules)
    print(f'Native corrected {args.device}: linear interpolation and gradient reproduced; Warp not imported')


if __name__ == '__main__':
    main()
