"""Public native preparation against the existing Warp CRK pipeline."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.crk import computeCRKFactors as warp_factors
from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend as warp_volume
from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend as warp_moments
from warpSPHCore.crk.crk_density import _computeCRKDensity_stateBackend as warp_density
from warpSPHCore.crk.native import (computeCRKVolume, computeCRKMoments, computeCRKDensity, prepareCRK)
from warpSPHCore.type_config import get_torch_precision


@pytest.fixture(scope='module', autouse=True)
def primal_modules():
    modules = ['warpSPHCore.crk.crk_volume', 'warpSPHCore.crk.crk_moments',
               'warpSPHCore.crk.crk_density']
    options = [wp.get_module_options(module=name) for name in modules]
    for name in modules: wp.set_module_options({'enable_backward': False}, module=name)
    yield
    for name, previous in zip(modules, options): wp.set_module_options(previous, module=name)


def particles(x):
    n = len(x)
    return sph.ParticleState(x, torch.linspace(.8, 1.5, n, device=x.device, dtype=x.dtype),
        x.new_full((n,), .2), (torch.arange(n, device=x.device) % 3).int())


def complete_csr(q, r):
    n, m = len(q.positions), len(r.positions)
    return sph.AdjacencyList(None, torch.arange(m, device=q.positions.device).repeat(n).to(sph.get_torch_index_dtype()),
        torch.full((n,), m, device=q.positions.device, dtype=torch.int32),
        torch.arange(n, device=q.positions.device, dtype=sph.get_torch_offset_dtype()) * m, n, m, numEdges=n*m)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('kernel', [sph.KernelFunctions.Wendland2, sph.KernelFunctions.CubicSpline,
                                   sph.KernelFunctions.Wendland4])
@pytest.mark.parametrize('direction', [sph.OperationDirection.TrueAllToToAll,
                                      sph.OperationDirection.FluidToBoundary])
def test_crk_passes_and_factors(device, dimension, kernel, direction):
    wp.init()
    axis = torch.tensor([-.3, 0., .3], device=device, dtype=get_torch_precision())
    r = particles(torch.cartesian_prod(*([axis] * dimension)).reshape(-1, dimension))
    q = particles(axis.new_tensor([[-.08] * dimension, [.1] * dimension]))
    # Periodicity includes an unwrapped query and a reference-image crossing.
    domain = sph.DomainDescription(axis.new_full((dimension,), -1), axis.new_ones(dimension),
        torch.tensor([True] + [False] * (dimension - 1), device=device), dimension)
    q.positions[:, 0] += 4
    graph = complete_csr(q, r)
    reference_graph = complete_csr(r, r)
    prop = sph.OperationProperties(kernel, operationMode=direction)
    tolerance = 2e-3 if axis.dtype == torch.float32 else 2e-9
    kwargs = dict(adjacency=graph, referenceParticles=r)
    volume = computeCRKVolume(q, prop, domain, **kwargs)
    torch.testing.assert_close(volume, warp_volume(q, prop, domain, **kwargs), rtol=tolerance, atol=tolerance)
    rv = computeCRKVolume(r, replace(prop, operationMode=sph.OperationDirection.TrueAllToToAll),
                          domain, reference_graph)
    prop = replace(prop, supportMode=sph.SupportScheme.Scatter)
    args = (q, prop, domain, volume, rv)
    actual = computeCRKMoments(*args, **kwargs)
    expected = warp_moments(*args, **kwargs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    native = prepareCRK(q, domain, kernel, direction, graph, r,
        referenceVolumeMode=sph.OperationDirection.TrueAllToToAll, referenceAdjacency=reference_graph)
    expected_volume, expected_density, state = warp_factors(q, domain, kernel, direction, graph, r,
        referenceVolumeMode=sph.OperationDirection.TrueAllToToAll)
    torch.testing.assert_close(native.queryVolumes, expected_volume, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(native.consistencyDensity, expected_density, rtol=tolerance, atol=tolerance)
    for name in ('A', 'B', 'gradA', 'gradB'):
        torch.testing.assert_close(getattr(native.crkState, name), getattr(state, name),
                                   rtol=tolerance, atol=tolerance)
    density = computeCRKDensity(q, prop, domain, state, volume, rv, **kwargs)
    torch.testing.assert_close(density, warp_density(q, prop, domain, state, volume, rv, **kwargs),
                               rtol=tolerance, atol=tolerance)
