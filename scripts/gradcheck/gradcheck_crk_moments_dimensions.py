#!/usr/bin/env python3
"""Finite-difference checks for all raw CRK moment outputs in 1D/2D/3D.

Use fixed topology and random scalar projections of all six floating-point
outputs. Geometry, support and volume adjoints are checked independently of
Warp's reverse pass, including tensor loops absent from the older 1D canary.
"""
import os
os.environ.setdefault('warpSPHCore_PRECISION', 'float64')

import torch
import warp as wp
from warpSPHCore import (
    DomainDescription, OperationProperties, ParticleState,
    radiusSearchCompactHashMap, setNeighborLanes,
)
from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
from warpSPHCore.enumTypes import KernelFunctions, SupportScheme


def check(dim, traversal):
    generator = torch.Generator().manual_seed(923 + dim)
    dtype, n = torch.float64, 7
    positions = (torch.rand(n, dim, generator=generator, dtype=dtype) - .5) * .6
    supports = torch.linspace(.9, 1.3, n, dtype=dtype)
    volumes = torch.linspace(.8, 1.2, n, dtype=dtype)
    masses = torch.ones(n, dtype=dtype)
    densities = torch.ones(n, dtype=dtype)
    kinds = torch.zeros(n, dtype=torch.int32)
    domain = DomainDescription(
        min=torch.full((dim,), -2., dtype=dtype),
        max=torch.full((dim,), 2., dtype=dtype),
        periodic=torch.ones(dim, dtype=torch.bool), dim=dim,
    )

    def particles(x, h):
        return ParticleState(positions=x, supports=h, masses=masses,
                             kinds=kinds, densities=densities)

    adjacency, grid = radiusSearchCompactHashMap(
        particles(positions, supports), domain, mode=SupportScheme.Gather,
        returnCompactHashMap=True,
    )
    topology = adjacency if traversal == 'adjacency' else grid
    properties = OperationProperties(kernel=KernelFunctions.Wendland2,
                                     supportMode=SupportScheme.Scatter)

    def raw(x, h, volume):
        return _computeCRKMoments_stateBackend(
            particles(x, h), properties, domain, adjacency=topology,
            queryVolumes=volume, referenceVolumes=volume,
        )[:6]

    initial = raw(positions, supports, volumes)
    weights = [torch.randn(t.shape, generator=generator, dtype=dtype) for t in initial]

    def projected(x, h, volume):
        return sum((value * weight).sum()
                   for value, weight in zip(raw(x, h, volume), weights))

    inputs = tuple(t.clone().requires_grad_(True) for t in (positions, supports, volumes))
    assert torch.autograd.gradcheck(projected, inputs, eps=1e-6, atol=2e-5,
                                    rtol=2e-4, fast_mode=True)
    print(f'moments: dim={dim}, traversal={traversal}: PASSED', flush=True)


if __name__ == '__main__':
    wp.init()
    torch.set_num_threads(2)
    setNeighborLanes(1)
    for dim in (1, 2, 3):
        for traversal in ('adjacency', 'grid'):
            check(dim, traversal)
