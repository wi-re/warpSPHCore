"""Corrected divergence/curl/Laplacian/covariance against readable Warp paths."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.native.api import sphOperation as native
from _support.operators import fixture


@pytest.fixture(scope='module', autouse=True)
def primal_reference_modules():
    modules = ['warpSPHCore.coreOperations.native.wp_' + name
               for name in ('divergence', 'curl', 'laplacian', 'covariance')]
    previous = [wp.get_module_options(module=name) for name in modules]
    for name in modules: wp.set_module_options({'enable_backward': False}, module=name)
    yield
    for name, options in zip(modules, previous): wp.set_module_options(options, module=name)


CASES = []
for operation in (sph.WarpOperation.Divergence, sph.WarpOperation.Curl,
                  sph.WarpOperation.Laplacian, sph.WarpOperation.Covariance):
    for rank in ([0] if operation == sph.WarpOperation.Covariance else [1, 2]):
        for gradient in sph.GradientScheme:
            for laplacian in (list(sph.LaplacianScheme) if operation == sph.WarpOperation.Laplacian
                              else [sph.LaplacianScheme.Brookshaw]):
                CASES.append((operation, rank, gradient, laplacian))


@pytest.mark.parametrize('operation,rank,gradient,laplacian', CASES)
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('support', [sph.SupportScheme.Gather, sph.SupportScheme.SuperSymmetric])
def test_remaining_matches_warp(device, operation, rank, gradient, laplacian, dimension, support):
    wp.init()
    dtype = sph.get_torch_precision()
    q, r, domain, fi, fj = fixture(device, dimension, rank)
    def cast(s):
        return replace(s, **{name: getattr(s, name).to(dtype) for name in
                            ('positions', 'supports', 'masses', 'densities')})
    q, r = cast(q), cast(r)
    q.positions[:, 0] += 4; r.positions[:, 0] -= 6
    domain = replace(domain, min=domain.min.to(dtype), max=domain.max.to(dtype))
    fi, fj = fi.to(dtype), fj.to(dtype)
    seed = torch.Generator().manual_seed(17821)
    def random(*size): return torch.rand(size, generator=seed, dtype=dtype).to(device)
    A = .7 + random(7)
    A[::2] *= -1  # Counts depend on the corrected value, not CSR degree.
    state = sph.CRKState(A, random(7, dimension) - .5,
                         random(7, dimension) - .5, random(7, dimension, dimension) - .5)
    R = sph.RenormalizationState(torch.eye(dimension, device=device, dtype=dtype)[None] +
                                .2 * random(7, dimension, dimension))
    options = dict(referenceParticles=r, referenceVolumes=.1 + random(5), crkState=state,
                   gradHState=sph.GradHState(.7 + random(7), .8 + random(5)), renormalizationState=R)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2 if support == sph.SupportScheme.Gather
                                        else sph.KernelFunctions.Wendland4,
        operation=operation, gradientMode=gradient, laplacianMode=laplacian,
        divergenceDotMode=bool(gradient.value % 2), supportMode=support,
        operationMode=sph.OperationDirection.FluidToBoundary if gradient.value % 2
                      else sph.OperationDirection.TrueAllToToAll)
    # Explicit broad CSR also exercises zero-weight admitted edges. The count
    # must inspect corrected kernel positivity, independently of reduction.
    ids = torch.arange(4, -1, -1, device=device, dtype=sph.get_torch_index_dtype()).repeat(7)
    graph = sph.AdjacencyList(None, torch.cat((ids.new_zeros(2), ids)),
        torch.full((7,), 5, device=device, dtype=torch.int32),
        torch.arange(7, device=device, dtype=sph.get_torch_offset_dtype()) * 5 + 2, 7, 5, numEdges=35)
    options['adjacency'] = graph
    if operation == sph.WarpOperation.Divergence: options['consistentDivergence'] = bool(rank % 2)
    if operation == sph.WarpOperation.Covariance: options['covarianceReturnNumNeighbors'] = True
    expected = sph.warpOperation(q, properties, domain, fi, fj, **options)
    actual = native(q, properties, domain, fi, fj, **options)
    a, b = (actual, expected) if isinstance(actual, tuple) else ((actual,), (expected,))
    tolerance = 8e-4 if dtype == torch.float32 else 4e-9
    for x, y in zip(a, b): torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)
