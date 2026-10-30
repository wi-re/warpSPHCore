"""Torch producers, Warp work and Torch consumers share one nondefault stream."""

from typing import Any

import pytest
import torch
import warp as wp

from warpSPHCore.autograd.launcher import launch_kernel
from warpSPHCore.autograd.stateAwareWarpFunction import StateAwareWarpFunction
from warpSPHCore.autograd.stateLessWarpFunction import warpWrapper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@wp.kernel
def square(x: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = x[i] * x[i]


def _square(x, bridge):
    if bridge == 'stateless':
        return warpWrapper(lambda a: launch_kernel(square, x.numel(), wp.dtype_from_torch(x.dtype), a), x)
    return StateAwareWarpFunction.apply(
        None, lambda arrays, **kw: tuple(arrays), launch_kernel, square,
        x.numel(), wp.dtype_from_torch(x.dtype), x)


@pytest.mark.parametrize('bridge', ['stateful', 'stateless'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_forward_backward_and_seed_ownership(bridge, dtype, monkeypatch):
    # Compile/allocate before the delayed producer; timing is not an assertion.
    _square(torch.ones(256, device='cuda', dtype=dtype, requires_grad=True), bridge).sum().backward()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    original = wp.launch
    launches = []

    def launch(*args, **kwargs):
        device = wp.get_device(kwargs.get('device'))
        actual = kwargs.get('stream') or wp.get_stream(device)
        launches.append((actual.cuda_stream, kwargs.get('adjoint', False)))
        assert actual.cuda_stream == side.cuda_stream
        return original(*args, **kwargs)

    monkeypatch.setattr(wp, 'launch', launch)
    with torch.cuda.stream(side):
        torch.cuda._sleep(20_000_000)
        x = torch.full((256,), 3., device='cuda', dtype=dtype, requires_grad=True)
        y = _square(x, bridge)
        seed = torch.full_like(y, 2.)
        g = torch.autograd.grad(y, x, seed)[0]
        answer = y + g
        # Reuse cached input views/adjoints, and retain the caller's seed.
        y2 = _square(x, bridge)
        g2 = torch.autograd.grad(y2, x, seed)[0]
    side.synchronize()
    assert any(adjoint for _, adjoint in launches)
    torch.testing.assert_close(answer, torch.full_like(answer, 21.))
    torch.testing.assert_close(g2, torch.full_like(g2, 12.))
    torch.testing.assert_close(seed, torch.full_like(seed, 2.))


@pytest.mark.parametrize('lanes', [1, 32])
def test_prepared_and_ordinary_density(particle_case, lanes):
    from dataclasses import replace
    from warpSPHCore import OperationProperties, setNeighborLanes, warpOperation
    from warpSPHCore.autograd.operator_spec import SPHContext
    from warpSPHCore.coreOperations.wp_density import _DENSITY_SPEC
    from warpSPHCore.enumTypes import KernelFunctions, SupportScheme, WarpOperation
    from warpSPHCore.mfm.fastLaunch import PreparedLauncher

    case = particle_case
    particles = case['particles']
    if not particles.positions.is_cuda:
        pytest.skip('requires CUDA particle case')
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather)
    expected = warpOperation(particles, props, case['domain'], adjacency=case['adjacency'])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    try:
        setNeighborLanes(lanes)
        with torch.cuda.stream(side):
            torch.cuda._sleep(20_000_000)
            p = replace(particles, masses=particles.masses * 3.)
            prepared = PreparedLauncher(SPHContext(query=p, properties=props, domain=case['domain'],
                                                   adjacency=case['adjacency']))
            ordinary = warpOperation(p, props, case['domain'], adjacency=case['adjacency'])
            fast = prepared.launch(_DENSITY_SPEC)
            consumer = fast + ordinary
        side.synchronize()
    finally:
        setNeighborLanes(None)
    torch.testing.assert_close(consumer, expected * 6.)


def test_value_jvp_on_nondefault_stream(particle_case):
    from conftest import op
    from warpSPHCore.enumTypes import WarpOperation

    if not particle_case['particles'].positions.is_cuda:
        pytest.skip('requires CUDA particle case')
    x = torch.ones_like(particle_case['particles'].masses)
    expected = op(particle_case, WarpOperation.Interpolate, reference_values=x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.autograd.forward_ad.dual_level():
        torch.cuda._sleep(20_000_000)
        dual = torch.autograd.forward_ad.make_dual(x * 3., x * 2.)
        output = op(particle_case, WarpOperation.Interpolate, reference_values=dual)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
        consumer = primal + tangent
    side.synchronize()
    torch.testing.assert_close(consumer, expected * 5.)


def test_geometry_jvp_and_density_hvp(particle_case, monkeypatch):
    from warpSPHCore import OperationProperties, ParticleTangentState, warpOperation, warpOperationJVP
    from warpSPHCore.coreOperations.wp_densityHVP import computeSPHDensityPositionHVP
    from warpSPHCore.enumTypes import KernelFunctions, SupportScheme, WarpOperation

    p, domain, adjacency = (particle_case[key] for key in ('particles', 'domain', 'adjacency'))
    if not p.positions.is_cuda:
        pytest.skip('requires CUDA particle case')
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather)
    expected = warpOperation(p, props, domain, adjacency=adjacency)
    tangent = p.positions.sin()
    hvp = lambda v: computeSPHDensityPositionHVP(p, domain, KernelFunctions.Wendland2,
                                                 SupportScheme.Gather, adjacency, v)
    expectedHvp = hvp(tangent)
    # Warm the geometry kernel separately from the delayed producer.
    warpOperationJVP(p, props, domain, adjacency=adjacency,
                     referenceTangentState=ParticleTangentState(positions=None, supports=None, masses=p.masses))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    original = wp.launch

    def launch(*args, **kwargs):
        actual = kwargs.get('stream') or wp.get_stream(kwargs.get('device'))
        assert actual.cuda_stream == side.cuda_stream
        return original(*args, **kwargs)

    monkeypatch.setattr(wp, 'launch', launch)
    with torch.cuda.stream(side):
        torch.cuda._sleep(20_000_000)
        jvp = warpOperationJVP(p, props, domain, adjacency=adjacency,
                               referenceTangentState=ParticleTangentState(positions=None, supports=None, masses=p.masses * 2))
        gotHvp = hvp(tangent * 2)
        consumer = jvp + gotHvp.sum(-1)
    side.synchronize()
    # HVP uses atomic pair sums; a different stream may change addition order.
    eps = torch.finfo(expected.dtype).eps
    bound = 128 * eps * (expected.abs().max() + expectedHvp.abs().max())
    torch.testing.assert_close(consumer, expected * 2 + expectedHvp.sum(-1) * 2,
                               rtol=128 * eps, atol=float(bound))


@pytest.mark.parametrize('bridge', ['stateful', 'stateless'])
def test_forward_graph_replays_changed_input(bridge):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        x = torch.ones(256, device='cuda')
        _square(x, bridge)  # module and view warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            y = _square(x, bridge)
        x.fill_(3.)
        graph.replay()
        first = y.clone()
        x.fill_(4.)
        graph.replay()
        second = y.clone()
    side.synchronize()
    torch.testing.assert_close(first, torch.full_like(first, 9.))
    torch.testing.assert_close(second, torch.full_like(second, 16.))
