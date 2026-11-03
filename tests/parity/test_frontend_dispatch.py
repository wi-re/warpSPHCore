"""The 0.7.1 Warp frontend can explicitly dispatch core work to C++."""
from dataclasses import replace
import importlib

import pytest
import torch
import warp as wp
import warpSPHCore as sph
from _support.operators import fixture


@pytest.fixture(scope='module', autouse=True)
def reference_runtime():
    wp.init()


def inputs(device):
    q, r, domain, fi, fj = fixture(device, 2, 1)
    dtype = sph.get_torch_precision()
    def cast(state):
        return replace(state, **{key: getattr(state, key).to(dtype)
                       for key in ('positions', 'supports', 'masses', 'densities')})
    q, r = cast(q), cast(r)
    domain = replace(domain, min=domain.min.to(dtype), max=domain.max.to(dtype))
    graph = sph.AdjacencyList(None,
        torch.arange(5, dtype=sph.get_torch_index_dtype(), device=device).repeat(7),
        torch.full((7,), 5, dtype=torch.int32, device=device),
        torch.arange(7, dtype=sph.get_torch_offset_dtype(), device=device) * 5,
        7, 5, numEdges=35)
    return q, r, domain, fi.to(dtype), fj.to(dtype), graph


def test_all_core_routes_and_calibrated_density(device, monkeypatch):
    q, r, domain, fi, fj, graph = inputs(device)
    module = importlib.import_module('warpSPHCore.operations')
    for operation in sph.WarpOperation:
        if operation == sph.WarpOperation.Custom:
            continue
        prop = sph.OperationProperties(sph.KernelFunctions.Wendland4,
            operation=operation, gradientMode=sph.GradientScheme.Difference)
        options = dict(referenceParticles=r, adjacency=graph)
        monkeypatch.setattr(module, 'OPERATORS', 'warp')
        expected = module.warpOperation(q, prop, domain, fi, fj, **options)
        monkeypatch.setattr(module, 'OPERATORS', 'native')
        actual = module.warpOperation(q, prop, domain, fi, fj, **options)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    prop = replace(prop, operation=sph.WarpOperation.Density,
                   calibrateNormalization=True, n_h=4.)
    monkeypatch.setattr(module, 'OPERATORS', 'warp')
    expected = module.warpOperation(q, prop, domain, **options)
    monkeypatch.setattr(module, 'OPERATORS', 'native')
    actual = module.warpOperation(q, prop, domain, **options)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    with pytest.raises(TypeError, match='CSR AdjacencyList'):
        module.warpOperation(q, prop, domain, adjacency=object())
    with pytest.raises(NotImplementedError, match='Warp-specific'):
        module.warpOperationJVP(q, prop, domain)
    with pytest.raises(NotImplementedError, match='second-order'):
        module.warpOperationHVP(q, prop, domain, q.positions)


def test_crk_accumulation_routes(device, monkeypatch):
    q, r, domain, _, _, graph = inputs(device)
    volume, moments, density = [importlib.import_module(f'warpSPHCore.crk.crk_{name}')
                               for name in ('volume', 'moments', 'density')]
    prop = sph.OperationProperties(sph.KernelFunctions.Wendland4)
    options = dict(referenceParticles=r, adjacency=graph)
    states = []
    for backend in ('warp', 'native'):
        for module in (volume, moments, density):
            monkeypatch.setattr(module, 'OPERATORS', backend)
        v = volume._computeCRKVolume_stateBackend(q, prop, domain, **options)
        rv = r.masses / r.densities
        m = moments._computeCRKMoments_stateBackend(q, prop, domain, v, rv, **options)
        coefficients = sph.CRKState(torch.ones_like(q.masses), q.positions * .1,
                                   torch.zeros_like(q.positions),
                                   q.positions.new_zeros(len(q.positions), 2, 2))
        rho = density._computeCRKDensity_stateBackend(q, prop, domain, coefficients, v, rv, **options)
        states.append((v, *m, rho))
    for expected, actual in zip(*states):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    policy = importlib.import_module('warpSPHCore._backend')
    monkeypatch.setattr(policy, 'OPERATORS', 'native')
    from warpSPHCore.crk import computeCRKFactorsJVP
    with pytest.raises(NotImplementedError, match='Warp-specific'):
        computeCRKFactorsJVP(q, domain, prop.kernel, None, adjacency=graph)
