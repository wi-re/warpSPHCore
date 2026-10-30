"""Optional reference-backend comparison, run separately per Warp precision."""
import os
from dataclasses import replace
import pytest
import torch

if os.environ.get('WARPSPHCORE_NATIVE_PARITY') != '1':
    pytest.skip('opt-in native/Warp cross-backend qualification', allow_module_level=True)

import warp as wp
import warpSPHCore as sph
from warpSPHCore.native.api import warpOperation as native_operation, radiusSearch


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('support', list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('direction', [sph.OperationDirection.TrueAllToToAll,
                                       sph.OperationDirection.AllToAll,
                                       sph.OperationDirection.BoundaryToFluid])
def test_native_matches_existing_warp(device, dim, support, direction):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('requires GPU')
    wp.init()
    dtype = torch.float64 if os.environ.get('warpSPHCore_PRECISION') == 'float64' else torch.float32
    gen = torch.Generator().manual_seed(331)
    def rand(*shape): return torch.rand(*shape, dtype=dtype, generator=gen).to(device)
    def particles(n):
        return sph.ParticleState(rand(n, dim) * 4 - 2, rand(n) + .2, rand(n) + .1,
                                 (torch.arange(n, device=device) % 4).int(), rand(n) + .7)
    q, r = particles(13), particles(8)
    d = sph.DomainDescription(q.positions.new_full((dim,), -1), q.positions.new_full((dim,), 1),
                              torch.tensor([True] + [False] * (dim - 1), device=device), dim)
    adjacency = radiusSearch(q, d, r, support)
    adjacency.paddedJ = adjacency.paddedJ.to(sph.get_torch_index_dtype())
    adjacency.edgeOffsets = adjacency.edgeOffsets.to(sph.get_torch_offset_dtype())
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=sph.WarpOperation.Density,
                                          supportMode=support, operationMode=direction)
    for op in (sph.WarpOperation.Density, sph.WarpOperation.Gradient):
        prop = replace(properties, operation=op, gradientMode=sph.GradientScheme.Difference)
        qf, rf = (None, None) if op == sph.WarpOperation.Density else (rand(13), rand(8))
        expected = sph.warpOperation(q, prop, d, qf, rf, adjacency=adjacency, referenceParticles=r)
        actual = native_operation(q, prop, d, qf, rf, adjacency=adjacency, referenceParticles=r)
        tol = 3e-5 if dtype == torch.float32 else 3e-12
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
