"""Trial lane kernels: traversal, schemes, corrections, AD and replay contracts."""

from dataclasses import replace

import pytest
import torch

from warpSPHCore import (DomainDescription, OperationProperties, ParticleState, ParticleTangentState,
                         radiusSearchCompactHashMap, setNeighborLanes, warpOperation, warpOperationJVP)
from warpSPHCore.dataTypes import CRKState, GradHState, RenormalizationState
from warpSPHCore.enumTypes import (GradientScheme, KernelFunctions, LaplacianScheme,
                                  OperationDirection, SupportScheme, WarpOperation)
from warpSPHCore.type_config import get_torch_precision


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='tiled kernels require CUDA')


@pytest.fixture(autouse=True)
def reset_lanes():
    yield
    setNeighborLanes(None)


def case(dim=2):
    dtype = get_torch_precision()
    coords = torch.linspace(-.4, .4, 3, device='cuda', dtype=dtype)
    grid = torch.meshgrid(*([coords] * dim), indexing='ij')
    positions = torch.stack([axis.flatten() for axis in grid], dim=1)
    positions = torch.cat((positions, torch.full((1, dim), 5., device='cuda', dtype=dtype)))
    n = positions.shape[0]
    supports = torch.linspace(.82, 1.06, n, device='cuda', dtype=dtype)
    p = ParticleState(positions=positions, supports=supports,
                      masses=torch.linspace(.8, 1.2, n, device='cuda', dtype=dtype),
                      densities=torch.linspace(1., 1.4, n, device='cuda', dtype=dtype),
                      kinds=torch.zeros(n, device='cuda', dtype=torch.int32))
    domain = DomainDescription(min=torch.full((dim,), -10., device='cuda', dtype=dtype),
                               max=torch.full((dim,), 10., device='cuda', dtype=dtype),
                               periodic=torch.zeros(dim, device='cuda', dtype=torch.bool), dim=dim)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.SuperSymmetric)
    return p, domain, adjacency


def close(a, b, *, derivative=False):
    assert a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    tolerance = (3e-10 if derivative else 3e-12) if a.dtype == torch.float64 else (6e-4 if derivative else 3e-5)
    scale = max(float(a.detach().abs().max()), float(b.detach().abs().max()), 1.)
    torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance * scale)


CONFIGS = [
    (WarpOperation.Curl, LaplacianScheme.Default, (2,)),
    (WarpOperation.Curl, LaplacianScheme.Default, (2, 2)),
    (WarpOperation.Laplacian, LaplacianScheme.Naive, ()),
    (WarpOperation.Laplacian, LaplacianScheme.Brookshaw, ()),
    (WarpOperation.Laplacian, LaplacianScheme.Default, ()),
    (WarpOperation.Laplacian, LaplacianScheme.Dot, (2,)),
    (WarpOperation.Laplacian, LaplacianScheme.Brookshaw, (2,)),
    (WarpOperation.Laplacian, LaplacianScheme.Default, (2, 2)),
]


def properties(operation, laplacian=LaplacianScheme.Brookshaw, gradient=GradientScheme.Difference, direction=OperationDirection.AllToAll):
    return OperationProperties(kernel=KernelFunctions.Wendland2, operation=operation,
                               gradientMode=gradient, laplacianMode=laplacian,
                               supportMode=SupportScheme.Gather, operationMode=direction)


def values(p, shape):
    generator = torch.Generator(device='cuda').manual_seed(17)
    q = torch.randn((p.positions.shape[0], *shape), generator=generator, device='cuda', dtype=p.positions.dtype)
    r = torch.randn(q.shape, generator=generator, device='cuda', dtype=q.dtype)
    return q, r


@pytest.mark.parametrize('lanes', [4, 16, 32])
@pytest.mark.parametrize('traversal', ['adjacency', 'grid'])
@pytest.mark.parametrize('gradient', list(GradientScheme))
@pytest.mark.parametrize('operation,laplacian,shape', CONFIGS)
def test_schemes_shapes_and_traversal(lanes, traversal, gradient, operation, laplacian, shape):
    p, domain, adjacency = case()
    q, r = values(p, shape)
    kw = dict(adjacency=adjacency if traversal == 'adjacency' else None, queryValues=q, referenceValues=r)
    props = properties(operation, laplacian, gradient)
    setNeighborLanes(1)
    expected = warpOperation(p, props, domain, **kw)
    setNeighborLanes(lanes)
    actual = warpOperation(p, props, domain, **kw)
    close(expected, actual)


@pytest.mark.parametrize('operation', [WarpOperation.Curl, WarpOperation.Laplacian])
@pytest.mark.parametrize('traversal', ['adjacency', 'grid'])
def test_filtered_and_empty_rows_write_zero(operation, traversal):
    p, domain, adjacency = case()
    p.kinds[0] = 1
    # The isolated last particle has no physical non-self pairs. In CSR,
    # remove its self entry as well to exercise a genuinely empty row.
    if traversal == 'adjacency':
        counts = adjacency.numNeighbors.clone()
        counts[-1] = 0
        adjacency = replace(adjacency, numNeighbors=counts)
    q, _ = values(p, (2,) if operation == WarpOperation.Curl else ())
    props = properties(operation, direction=OperationDirection.FluidToFluid)
    setNeighborLanes(32)
    out = warpOperation(p, props, domain, adjacency=adjacency if traversal == 'adjacency' else None,
                         queryValues=q, referenceValues=q)
    assert torch.count_nonzero(out[0]) == 0
    assert torch.count_nonzero(out[-1]) == 0


@pytest.mark.parametrize('operation', [WarpOperation.Curl, WarpOperation.Laplacian])
@pytest.mark.parametrize('traversal', ['adjacency', 'grid'])
@pytest.mark.parametrize('correction', ['none', 'crk', 'renorm', 'grad_h', 'volumes'])
def test_reverse_geometry_values_and_corrections(operation, traversal, correction):
    p, domain, adjacency = case()
    q, r = values(p, (2,) if operation == WarpOperation.Curl else ())
    props = properties(operation)

    def run(lanes):
        setNeighborLanes(lanes)
        inputs = [t.clone().requires_grad_() for t in (p.positions, p.supports, p.masses, p.densities, q, r)]
        pp = replace(p, positions=inputs[0], supports=inputs[1], masses=inputs[2], densities=inputs[3])
        n = q.shape[0]
        extra = {}
        if correction == 'crk':
            fields = [torch.full((n,), 1.1, device=q.device, dtype=q.dtype),
                      torch.full((n, 2), .03, device=q.device, dtype=q.dtype),
                      torch.full((n, 2), .02, device=q.device, dtype=q.dtype),
                      torch.full((n, 2, 2), .01, device=q.device, dtype=q.dtype)]
            fields = [t.requires_grad_() for t in fields]
            inputs += fields
            extra['crkState'] = CRKState(A=fields[0], B=fields[1], gradA=fields[2], gradB=fields[3])
        elif correction == 'renorm':
            matrix = (torch.eye(2, device=q.device, dtype=q.dtype).expand(n, 2, 2).clone() * 1.2).requires_grad_()
            inputs.append(matrix)
            extra['renormalizationState'] = RenormalizationState(renormalizationMatrices=matrix)
        elif correction == 'grad_h':
            omega = torch.full_like(p.supports, .9, requires_grad=True)
            inputs.append(omega)
            extra['gradHState'] = GradHState(queryOmegas=omega, referenceOmegas=omega)
        elif correction == 'volumes':
            volume = torch.full_like(p.masses, .8, requires_grad=True)
            inputs.append(volume)
            extra['queryVolumes'] = extra['referenceVolumes'] = volume
        out = warpOperation(pp, props, domain, adjacency=adjacency if traversal == 'adjacency' else None,
                             queryValues=inputs[4], referenceValues=inputs[5], **extra)
        seed = torch.linspace(-.7, .8, out.numel(), device=q.device, dtype=q.dtype).reshape_as(out)
        grads = torch.autograd.grad(out, inputs, seed, allow_unused=True)
        return out, grads

    expected, reference = run(1)
    actual, grads = run(32)
    close(expected, actual)
    for a, b in zip(reference, grads):
        assert (a is None) == (b is None)
        if a is not None:
            close(a, b, derivative=True)


@pytest.mark.parametrize('operation', [WarpOperation.Curl, WarpOperation.Laplacian])
@pytest.mark.parametrize('traversal', ['adjacency', 'grid'])
def test_combined_jvp_vjp_duality(operation, traversal):
    p, domain, adjacency = case()
    q, r = values(p, (2,) if operation == WarpOperation.Curl else ())
    inputs = [t.clone().requires_grad_() for t in (p.positions, p.supports, p.masses, p.densities, q, r)]
    pp = replace(p, positions=inputs[0], supports=inputs[1], masses=inputs[2], densities=inputs[3])
    directions = [t.detach().sin() * .13 for t in inputs]
    props = properties(operation)
    adj = adjacency if traversal == 'adjacency' else None
    setNeighborLanes(32)
    out = warpOperation(pp, props, domain, adjacency=adj, queryValues=inputs[4], referenceValues=inputs[5])
    seed = torch.linspace(-.7, .8, out.numel(), device=q.device, dtype=q.dtype).reshape_as(out)
    grads = torch.autograd.grad(out, inputs, seed)
    tangent = warpOperationJVP(p, props, domain, adjacency=adj, queryValues=q, referenceValues=r,
                               queryTangentState=ParticleTangentState(positions=directions[0], supports=directions[1], masses=None,
                                                                     densities=directions[3]),
                               referenceTangentState=ParticleTangentState(positions=directions[0], supports=directions[1],
                                                                         masses=directions[2], densities=directions[3]),
                               tangentQueryValues=directions[4], tangentReferenceValues=directions[5])
    lhs = (seed * tangent).sum().reshape(1)
    rhs = sum((grad * direction).sum() for grad, direction in zip(grads, directions)).reshape(1)
    close(lhs, rhs, derivative=True)


@pytest.mark.parametrize('dim', [1, 3])
@pytest.mark.parametrize('operation', [WarpOperation.Curl, WarpOperation.Laplacian])
def test_other_spatial_dimensions(dim, operation):
    p, domain, adjacency = case(dim)
    q, r = values(p, (dim,))
    props = properties(operation)
    setNeighborLanes(1)
    expected = warpOperation(p, props, domain, adjacency=adjacency, queryValues=q, referenceValues=r)
    setNeighborLanes(32)
    actual = warpOperation(p, props, domain, adjacency=adjacency, queryValues=q, referenceValues=r)
    close(expected, actual)
    if dim == 1 and operation == WarpOperation.Curl:
        assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize('operation', [WarpOperation.Curl, WarpOperation.Laplacian])
def test_dual_value_wrapper(operation):
    p, domain, adjacency = case()
    q, r = values(p, (2,) if operation == WarpOperation.Curl else ())
    dq, dr = q.sin() * .1, r.cos() * .1
    props = properties(operation)
    setNeighborLanes(32)
    expected = warpOperation(p, props, domain, adjacency=adjacency, queryValues=dq, referenceValues=dr)
    with torch.autograd.forward_ad.dual_level():
        out = warpOperation(p, props, domain, adjacency=adjacency,
                             queryValues=torch.autograd.forward_ad.make_dual(q, dq),
                             referenceValues=torch.autograd.forward_ad.make_dual(r, dr))
        _, tangent = torch.autograd.forward_ad.unpack_dual(out)
        assert tangent is not None
        close(expected, tangent)


@pytest.mark.parametrize('lanes', [None, 4, 32])
@pytest.mark.parametrize('operation', [WarpOperation.Curl, WarpOperation.Laplacian])
def test_prepared_nondefault_stream_capture_replay(operation, lanes, monkeypatch):
    import warp as wp
    from warpSPHCore.autograd.operator_spec import SPHContext
    from warpSPHCore.autograd.lanes import neighborLanes
    from warpSPHCore.autograd.stream import torchWarpCapture, torchWarpStream
    from warpSPHCore.coreOperations.wp_curl import _CURL_SPEC
    from warpSPHCore.coreOperations.wp_laplacian import _LAPLACIAN_SPEC
    from warpSPHCore.mfm.fastLaunch import PreparedLauncher

    p, domain, adjacency = case()
    spec = _CURL_SPEC if operation == WarpOperation.Curl else _LAPLACIAN_SPEC
    shape = (2,) if operation == WarpOperation.Curl else ()
    q, r = values(p, shape)
    props = properties(operation)
    setNeighborLanes(lanes)
    expected = warpOperation(p, props, domain, adjacency=adjacency, queryValues=q, referenceValues=r)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    original = wp.launch
    launched = []

    def observe(*args, **kwargs):
        kernel = kwargs.get('kernel', args[0] if args else None)
        if kernel in (spec.kernel, spec.tiledKernel):
            launched.append(kernel)
        return original(*args, **kwargs)

    monkeypatch.setattr(wp, 'launch', observe)
    with torch.cuda.stream(stream), torchWarpStream(q.device):
        prepared = PreparedLauncher(SPHContext(query=p, properties=props, domain=domain, adjacency=adjacency))
        extras = dict(numDims=0 if not shape else 1, flatInputShape=1 if not shape else 2, flatOutputShape=1,
                      queryValuesFlat=q.view(q.shape[0], -1), referenceValuesFlat=r.view(r.shape[0], -1))
        snapshot = prepared.launch(spec, **extras)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torchWarpCapture(graph):
            out = prepared.launch(spec, **extras)
        graph.replay()
        consumer = out.clone()
        # Same storage, changed contents must be visible without rebuilding.
        q.mul_(2)
        r.mul_(2)
        graph.replay()
        changed = out.clone()
    stream.synchronize()
    close(snapshot.reshape_as(expected), expected)
    close(consumer.reshape_as(expected), expected)
    close(changed.reshape_as(expected), expected * 2)
    chosen = spec.tiledKernel if neighborLanes(default=spec.defaultNeighborLanes) > 1 else spec.kernel
    assert launched and all(kernel is chosen for kernel in launched)
