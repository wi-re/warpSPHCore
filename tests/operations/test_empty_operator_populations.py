"""Empty searches compose with operator values, derivatives and corrections."""

from dataclasses import replace

import pytest
import torch
import warp as wp

from warpSPHCore import (DomainDescription, ParticleState, ParticleTangentState, OperationProperties,
    WarpOperation, KernelFunctions, GradientScheme, LaplacianScheme, SupportScheme,
    radiusSearchCompactHashMap, warpOperation, warpOperationJVP, warpOperationHVP)
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.renorm import computeRenormalizationMatricesJVP
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.autograd.stream import torchWarpCapture, torchWarpStream


CASES = [(op, None) for op in (WarpOperation.Density, WarpOperation.Interpolate,
                              WarpOperation.Gradient, WarpOperation.Divergence,
                              WarpOperation.Curl, WarpOperation.Covariance)] + [
    (WarpOperation.Laplacian, lap) for lap in LaplacianScheme]


def setup(device, sizes, dim=2):
    dtype = get_torch_precision()

    def state(n):
        positions = torch.arange(n * dim, dtype=dtype, device=device).reshape(n, dim).cos() * .3
        scalar = torch.ones(n, dtype=dtype, device=device)
        return ParticleState(positions.requires_grad_(), (scalar * 1.3).requires_grad_(),
                              scalar.clone().requires_grad_(),
                              torch.zeros(n, dtype=torch.int32, device=device), scalar.clone().requires_grad_())

    q, r = state(sizes[0]), state(sizes[1])
    domain = DomainDescription(q.positions.new_full((dim,), -2), q.positions.new_full((dim,), 2),
                               torch.zeros(dim, dtype=torch.bool, device=device), dim)
    adj = radiusSearchCompactHashMap(q, domain, referenceParticles=r, mode=SupportScheme.SuperSymmetric)
    return q, r, domain, adj


@pytest.mark.parametrize('sizes', [(0, 5), (3, 0), (0, 0)])
@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
@pytest.mark.parametrize('operation,laplacian', CASES)
def test_empty_values_geometry_and_reverse_ad(device, sizes, traversal, operation, laplacian):
    q, r, domain, adjacency = setup(device, sizes)
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    vector = operation in (WarpOperation.Divergence, WarpOperation.Curl) or laplacian == LaplacianScheme.Dot
    fq = q.positions.detach().clone() if vector else q.masses.detach().clone()
    fr = r.positions.detach().clone() if vector else r.masses.detach().clone()
    fq.requires_grad_(), fr.requires_grad_()
    props = OperationProperties(operation=operation, kernel=KernelFunctions.Wendland2,
                                gradientMode=GradientScheme.Symmetric,
                                laplacianMode=laplacian or LaplacianScheme.Brookshaw)
    kwargs = dict(adjacency=adj, referenceParticles=r)
    jvp_kwargs = {}
    inputs = [q.positions, r.positions, q.supports, r.supports, q.masses, r.masses, q.densities, r.densities]
    if operation not in (WarpOperation.Density, WarpOperation.Covariance):
        kwargs['referenceValues'] = fr
        jvp_kwargs['tangentReferenceValues'] = torch.ones_like(fr)
        inputs.append(fr)
        if operation != WarpOperation.Interpolate:
            kwargs['queryValues'] = fq
            jvp_kwargs['tangentQueryValues'] = torch.ones_like(fq)
            inputs.append(fq)
    tq = ParticleTangentState(torch.ones_like(q.positions), torch.ones_like(q.supports), None,
                              None if operation == WarpOperation.Density else torch.ones_like(q.densities))
    tr = ParticleTangentState(torch.ones_like(r.positions), torch.ones_like(r.supports), torch.ones_like(r.masses),
                              None if operation == WarpOperation.Density else torch.ones_like(r.densities))
    result = warpOperation(q, props, domain, **kwargs)
    tangent = warpOperationJVP(q, props, domain, **kwargs, **jvp_kwargs,
                               queryTangentState=tq, referenceTangentState=tr)
    assert result.shape[0] == sizes[0] and result.shape == tangent.shape
    assert torch.isfinite(result).all() and torch.count_nonzero(result) == 0
    assert torch.isfinite(tangent).all() and torch.count_nonzero(tangent) == 0
    for output in (result, tangent):
        gradients = torch.autograd.grad(output.sum(), inputs, allow_unused=True)
        for grad in gradients:
            if grad is not None:
                assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0


@pytest.mark.parametrize('sizes', [(0, 5), (3, 0), (0, 0), (3, 5)])
@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
def test_empty_correction_constructors(device, sizes, traversal):
    q, r, domain, adjacency = setup(device, sizes)
    if all(sizes):
        # Nonempty populations with no contributing cross-population pairs.
        q = replace(q, positions=(q.positions.detach() - .8).requires_grad_(),
                    supports=(q.supports.detach() * .1).requires_grad_())
        r = replace(r, positions=(r.positions.detach() + .8).requires_grad_(),
                    supports=(r.supports.detach() * .1).requires_grad_())
        adjacency = radiusSearchCompactHashMap(q, domain, referenceParticles=r)
        assert adjacency.numEdges == 0
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    tq = ParticleTangentState(torch.ones_like(q.positions), torch.ones_like(q.supports), None)
    tr = ParticleTangentState(torch.ones_like(r.positions), torch.ones_like(r.supports), None)
    props = OperationProperties(operation=WarpOperation.Gradient, kernel=KernelFunctions.Wendland2)
    C, eig, state, tangent = computeRenormalizationMatricesJVP(q, props, domain, tq,
        referenceParticles=r, referenceTangentState=tr, adjacency=adj)
    if sizes[1] == 0:
        identity = torch.eye(2, dtype=q.positions.dtype, device=device).expand(sizes[0], 2, 2)
        torch.testing.assert_close(C, identity, rtol=0, atol=0)
        torch.testing.assert_close(state.renormalizationMatrices, identity, rtol=0, atol=0)
    assert torch.count_nonzero(tangent.renormalizationMatrices) == 0
    volume, density, correction = computeCRKFactors(q, domain, KernelFunctions.Wendland2,
                                                   referenceState=r, adjacency=adj)
    v, dv, factors, dfactors = computeCRKFactorsJVP(q, domain, KernelFunctions.Wendland2, tq,
        referenceState=r, referenceTangentState=tr, adjacency=adj)
    for value in (volume, density, v, dv, *vars(correction).values(), *vars(factors).values(), *vars(dfactors).values()):
        assert value.shape[0] == sizes[0] and torch.isfinite(value).all()
    for value in (volume, density, v, dv, *vars(dfactors).values()):
        assert torch.count_nonzero(value) == 0
    for value in (volume, density, *vars(correction).values()):
        grads = torch.autograd.grad(value.sum(), (q.positions, r.positions, q.supports, r.supports),
                                    allow_unused=True, retain_graph=True)
        assert all(g is None or (torch.isfinite(g).all() and torch.count_nonzero(g) == 0) for g in grads)


@pytest.mark.parametrize('sizes', [(0, 5), (3, 0), (0, 0)])
@pytest.mark.parametrize('field_shape', [(2, 2), (2, 2, 2)])
def test_empty_interpolation_preserves_tensor_field_axes(device, sizes, field_shape):
    q, r, domain, adj = setup(device, sizes)
    field = torch.ones((sizes[1], *field_shape), dtype=q.positions.dtype, device=device, requires_grad=True)
    result = warpOperation(q, OperationProperties(operation=WarpOperation.Interpolate,
        kernel=KernelFunctions.Wendland2), domain, referenceParticles=r, adjacency=adj, referenceValues=field)
    assert result.shape == (sizes[0], *field_shape)
    assert torch.count_nonzero(result) == 0
    grad, = torch.autograd.grad(result.sum(), (field,))
    assert grad.shape == field.shape and torch.count_nonzero(grad) == 0


@pytest.mark.parametrize('sizes', [(0, 5), (3, 0), (0, 0)])
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_empty_density_dual_and_position_hvp(device, sizes, dim):
    from torch.autograd import forward_ad
    if dim == 3 and device.type == 'cpu':
        pytest.skip('3D SPH qualification runs on CUDA')
    q, r, domain, adj = setup(device, sizes, dim)
    props = OperationProperties(operation=WarpOperation.Density, kernel=KernelFunctions.Wendland2)
    with forward_ad.dual_level():
        query = replace(q, positions=forward_ad.make_dual(q.positions.detach(), torch.ones_like(q.positions)))
        ref = replace(r, positions=forward_ad.make_dual(r.positions.detach(), torch.ones_like(r.positions)))
        primal, tangent = forward_ad.unpack_dual(warpOperation(query, props, domain,
            referenceParticles=ref, adjacency=adj))
        assert primal.shape == tangent.shape == (sizes[0],)
        assert torch.count_nonzero(primal) == torch.count_nonzero(tangent) == 0
    hvp = warpOperationHVP(q, props, domain, torch.ones_like(q.positions), adjacency=adj,
        referenceParticles=r, tangentReferencePositions=torch.ones_like(r.positions))
    assert hvp.shape == q.positions.shape and torch.count_nonzero(hvp) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA capture/replay')
@pytest.mark.parametrize('sizes', [(0, 5), (3, 0), (0, 0)])
@pytest.mark.parametrize('traversal', ['csr', 'grid'])
def test_empty_density_and_jvp_current_stream_and_capture(sizes, traversal):
    wp.init()
    q, r, domain, adjacency = setup(torch.device('cuda'), sizes)
    adj = adjacency if traversal == 'csr' else adjacency.hashMap
    props = OperationProperties(operation=WarpOperation.Density, kernel=KernelFunctions.Wendland2)
    tq = ParticleTangentState(torch.ones_like(q.positions), None, None)
    tr = ParticleTangentState(torch.ones_like(r.positions), None, None)

    def evaluate():
        value = warpOperation(q, props, domain, adjacency=adj, referenceParticles=r)
        tangent = warpOperationJVP(q, props, domain, adjacency=adj, referenceParticles=r,
            queryTangentState=tq, referenceTangentState=tr)
        return value, tangent

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate()
    graph = torch.cuda.CUDAGraph()
    # Older taped calls can leave cycles owning Warp resources. The shared
    # capture boundary defers their GC until after capture and retains Warp's
    # external-capture resources, as in the native solver capture paths.
    with torch.cuda.stream(stream), torchWarpStream(q.positions.device):
        with torchWarpCapture(graph):
            outputs = evaluate()
    for _ in range(3):
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    for value in outputs:
        assert value.shape == (sizes[0],) and torch.isfinite(value).all()
        assert torch.count_nonzero(value) == 0
