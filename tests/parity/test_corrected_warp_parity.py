"""Corrected native primals versus the existing Warp implementation."""
from dataclasses import replace
import os

import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.native.api import sphOperation as native, radiusSearch
from _support.operators import fixture


@pytest.fixture(scope='module', autouse=True)
def primal_reference_modules():
    modules = ['warpSPHCore.coreOperations.native.wp_interpolate',
               'warpSPHCore.coreOperations.native.wp_gradient']
    previous = [wp.get_module_options(module=name) for name in modules]
    for name in modules:wp.set_module_options({'enable_backward': False}, module=name)
    yield
    for name, options in zip(modules, previous):wp.set_module_options(options, module=name)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('scheme', list(sph.GradientScheme))
@pytest.mark.parametrize('support', [sph.SupportScheme.Gather, sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('operation', [sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient])
def test_corrected_primal_matches_warp(device, dimension, rank, scheme, support, operation):
    wp.init()
    dtype = torch.float64 if os.environ.get('warpSPHCore_PRECISION') == 'float64' else torch.float32
    q, r, domain, qf, rf = fixture(device, dimension, rank)
    def cast(s):return replace(s, **{name: getattr(s, name).to(dtype) for name in ('positions','supports','masses','densities')})
    q, r = cast(q), cast(r)
    domain = replace(domain, min=domain.min.to(dtype), max=domain.max.to(dtype))
    qf, rf = qf.to(dtype), rf.to(dtype)
    generator = torch.Generator().manual_seed(671)
    def random(*size):return torch.rand(size, generator=generator, dtype=dtype).to(device)
    crk = sph.CRKState(.7 + random(7), random(7,dimension) - .5,
                      random(7,dimension) - .5, random(7,dimension,dimension) - .5)
    kw = dict(queryVolumes=.1 + random(7), referenceVolumes=.1 + random(5), crkState=crk)
    if operation == sph.WarpOperation.Gradient:
        kw.update(gradHState=sph.GradHState(.7 + random(7), .8 + random(5)),
            renormalizationState=sph.RenormalizationState(torch.eye(dimension, device=device, dtype=dtype)[None] + .2 * random(7,dimension,dimension)))
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland2 if support == sph.SupportScheme.Gather
                                  else sph.KernelFunctions.Wendland4,
        operation=operation, gradientMode=scheme, supportMode=support,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    graph = radiusSearch(q, domain, r, support)
    graph.paddedJ = graph.paddedJ.to(sph.get_torch_index_dtype())
    graph.edgeOffsets = graph.edgeOffsets.to(sph.get_torch_offset_dtype())
    expected = sph.warpOperation(q, prop, domain, qf, rf, referenceParticles=r, adjacency=graph, **kw)
    actual = native(q, prop, domain, qf, rf, referenceParticles=r, adjacency=graph, **kw)
    tolerance = 4e-4 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
