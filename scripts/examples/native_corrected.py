"""Warp-free CRK preparation and corrected core operators.

Run WARPSPHCORE_BACKEND=native python scripts/examples/native_corrected.py
--device cuda (or cpu). Particle initialization and the small batched factor
solve use Torch; all particle-pair passes use native CPU/CUDA/HIP kernels.
"""
import argparse
from dataclasses import replace
import sys

import torch
import warpSPHCore as sph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if sph.backend != 'native':
        raise RuntimeError('Set WARPSPHCORE_BACKEND=native before importing the package')

    # Keep the example small enough to inspect. Queries and references are
    # distinct populations, so their apparent-volume estimates differ.
    axis = torch.tensor([-.4, 0., .4], dtype=torch.float64, device=args.device)
    reference_positions = torch.cartesian_prod(axis, axis)
    query_positions = axis.new_tensor([[-.1, .08], [.12, -.06]])

    def particles(positions):
        return sph.ParticleState(
            positions=positions,
            supports=positions.new_full((len(positions),), 1.3),
            masses=positions.new_full((len(positions),), .2),
            kinds=torch.zeros(len(positions), device=args.device, dtype=torch.int32))

    query, reference = particles(query_positions), particles(reference_positions)
    domain = sph.DomainDescription(
        min=axis.new_full((2,), -1), max=axis.new_ones(2),
        periodic=torch.zeros(2, device=args.device, dtype=torch.bool), dim=2)
    kernel = sph.KernelFunctions.Wendland2
    direction = sph.OperationDirection.TrueAllToToAll

    # The preparation wrapper accepts existing CSR lists from a frontend.
    # With unequal supports, use a union graph to cover Gather and Scatter.
    graph = sph.radiusSearch(query, domain, reference, sph.SupportScheme.SuperSymmetric)
    reference_graph = sph.radiusSearch(reference, domain)
    preparation = sph.prepareCRK(
        query, domain, kernel, direction, graph, reference,
        referenceAdjacency=reference_graph)

    # Both operators consume the existing CRKState and explicit reference
    # volumes. No coefficient repacking or particle permutation is needed.
    query_values = 1 + 2 * query.positions[:, 0] - 3 * query.positions[:, 1]
    reference_values = 1 + 2 * reference.positions[:, 0] - 3 * reference.positions[:, 1]
    properties = sph.OperationProperties(kernel, supportMode=sph.SupportScheme.Scatter,
                                          operationMode=direction)
    options = dict(referenceParticles=reference, adjacency=graph,
                   referenceVolumes=preparation.referenceVolumes, crkState=preparation.crkState)
    interpolation = sph.warpOperation(query, properties, domain, query_values, reference_values, **options)
    gradient = sph.warpOperation(query, replace(properties, operation=sph.WarpOperation.Gradient),
                                 domain, query_values, reference_values, **options)
    torch.testing.assert_close(interpolation, query_values, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(gradient, gradient.new_tensor([2., -3.]).expand_as(gradient),
                               rtol=2e-12, atol=2e-12)
    # The same prepared state drives vector contractions and geometric
    # covariance. This affine velocity has divergence 6 and 2D curl 4.
    def velocity(positions):
        x, y = positions.unbind(dim=1)
        return torch.stack((2 * x - y, 3 * x + 4 * y), dim=1)

    query_velocity, reference_velocity = velocity(query.positions), velocity(reference.positions)
    divergence = sph.warpOperation(query, replace(properties, operation=sph.WarpOperation.Divergence),
        domain, query_velocity, reference_velocity, **options)
    curl = sph.warpOperation(query, replace(properties, operation=sph.WarpOperation.Curl),
        domain, query_velocity, reference_velocity, **options)
    covariance, counts = sph.warpOperation(query, replace(properties, operation=sph.WarpOperation.Covariance),
        domain, covarianceReturnNumNeighbors=True, **options)
    # A constant field has zero Laplacian under all four gradient weightings.
    constant_query, constant_reference = torch.ones_like(query_values), torch.ones_like(reference_values)
    laplacian = sph.warpOperation(query, replace(properties, operation=sph.WarpOperation.Laplacian),
        domain, constant_query, constant_reference, **options)
    torch.testing.assert_close(divergence, torch.full_like(divergence, 6.), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(curl, torch.full_like(curl, 4.), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(covariance, torch.eye(2).to(covariance).expand_as(covariance),
                               rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(laplacian, torch.zeros_like(laplacian), rtol=0, atol=0)
    assert counts.shape == (len(query.positions),)
    assert not any(name == 'warp' or name.startswith('warp.') for name in sys.modules)
    print(f'Native CRK {args.device}: preparation and all six corrected core primals passed; Warp not imported')


if __name__ == '__main__':
    main()
