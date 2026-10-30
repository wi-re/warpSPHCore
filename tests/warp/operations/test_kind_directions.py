"""Frozen interaction rules must agree between primal and derivative paths."""

from dataclasses import replace

import pytest
import torch

from warpSPHCore import (
    DomainDescription, GradientScheme, KernelFunctions, LaplacianScheme,
    OperationDirection, OperationProperties, ParticleState, ParticleTangentState,
    SupportScheme, WarpOperation, AdjacencyList, radiusSearchCompactHashMap,
    warpOperation, warpOperationJVP, warpOperationHVP,
)
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP
from warpSPHCore.crk.crk_terms import computeCRKTermsWarp
from warpSPHCore.type_config import get_torch_offset_dtype


CASES = [(op, None) for op in (WarpOperation.Density, WarpOperation.Interpolate,
                              WarpOperation.Gradient, WarpOperation.Divergence,
                              WarpOperation.Curl, WarpOperation.Covariance)] + [
    (WarpOperation.Laplacian, scheme) for scheme in LaplacianScheme]


def cloud(device):
    dtype = get_torch_precision()
    q = torch.tensor([[-.3, -.2], [.25, -.3], [-.2, .3], [.3, .2]], dtype=dtype, device=device)
    r = torch.tensor([[.1, -.2], [-.25, .15], [.25, .3], [-.1, -.3], [.35, .05]],
                     dtype=dtype, device=device)

    def state(pos, kinds):
        n = len(pos)
        return ParticleState(pos, torch.linspace(1.2, 1.4, n, dtype=dtype, device=device),
                             torch.linspace(.8, 1.1, n, dtype=dtype, device=device),
                             torch.tensor(kinds, dtype=torch.int32, device=device),
                             torch.linspace(.9, 1.3, n, dtype=dtype, device=device))

    query, reference = state(q, [0, 1, 2, 3]), state(r, [2, 0, 3, 1, 0])
    domain = DomainDescription(q.new_full((2,), -2), q.new_full((2,), 2),
                               torch.zeros(2, dtype=torch.bool, device=device), 2)
    adjacency = radiusSearchCompactHashMap(query, domain, referenceParticles=reference,
                                          mode=SupportScheme.SuperSymmetric)
    return query, reference, domain, adjacency


@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
@pytest.mark.parametrize('direction', list(OperationDirection))
@pytest.mark.parametrize('operation,laplacian', CASES)
def test_mixed_kind_combined_jvp_matches_masked_primal(device, traversal, direction, operation, laplacian):
    q, r, domain, adjacency = cloud(device)
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    q = replace(q, positions=q.positions.requires_grad_(), supports=q.supports.requires_grad_(),
                densities=q.densities.requires_grad_())
    r = replace(r, positions=r.positions.requires_grad_(), supports=r.supports.requires_grad_(),
                masses=r.masses.requires_grad_(), densities=r.densities.requires_grad_())
    tq = ParticleTangentState(q.positions.cos() * .07, q.supports * .03, None, q.densities * -.02)
    tr = ParticleTangentState(r.positions.sin() * .11, r.supports * -.04, r.masses * .06, r.densities * .025)
    vector = operation in (WarpOperation.Divergence, WarpOperation.Curl) or laplacian == LaplacianScheme.Dot
    fq = (q.positions.sin() if vector else q.positions[:, 0].sin()).detach().requires_grad_()
    fr = (r.positions.cos() if vector else r.positions[:, 1].cos()).detach().requires_grad_()
    dfq, dfr = fq.detach().cos() * .04, fr.detach().sin() * .05
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=operation,
                                operationMode=direction, supportMode=SupportScheme.Gather,
                                gradientMode=GradientScheme.Symmetric,
                                laplacianMode=laplacian or LaplacianScheme.Brookshaw)
    inputs = [q.positions, r.positions, q.supports, r.supports, r.masses, q.densities, r.densities]
    tangents = [tq.positions, tr.positions, tq.supports, tr.supports, tr.masses, tq.densities, tr.densities]
    kwargs = dict(adjacency=adj, referenceParticles=r)
    jvp_kwargs = {}
    if operation not in (WarpOperation.Density, WarpOperation.Covariance):
        kwargs.update(referenceValues=fr)
        jvp_kwargs.update(tangentReferenceValues=dfr)
        inputs.append(fr)
        tangents.append(dfr)
        if operation != WarpOperation.Interpolate:
            kwargs.update(queryValues=fq)
            jvp_kwargs.update(tangentQueryValues=dfq)
            inputs.append(fq)
            tangents.append(dfq)
    if operation == WarpOperation.Density:
        tq, tr = replace(tq, densities=None), replace(tr, densities=None)
    output = warpOperation(q, props, domain, **kwargs)
    cotangent = torch.arange(output.numel(), dtype=output.dtype, device=device).reshape_as(output).cos()
    vjp = torch.autograd.grad((output * cotangent).sum(), inputs, allow_unused=True)
    expected = sum((g * v).sum() for g, v in zip(vjp, tangents) if g is not None)
    actual = warpOperationJVP(q, props, domain, **kwargs, **jvp_kwargs,
                              queryTangentState=tq, referenceTangentState=tr)
    tol = 2e-4 if output.dtype == torch.float32 else 2e-9
    torch.testing.assert_close((actual * cotangent).sum(), expected, rtol=tol, atol=tol)


def test_excluded_ghost_mass_tangent_is_exactly_zero(device):
    q, r, domain, adjacency = cloud(device)
    zero = ParticleTangentState(torch.zeros_like(q.positions), None, None)
    ghost_only = torch.zeros_like(r.masses)
    ghost_only[r.kinds == 2] = 1
    ref = ParticleTangentState(torch.zeros_like(r.positions), None, ghost_only)
    result = warpOperationJVP(q, OperationProperties(operation=WarpOperation.Density,
                              kernel=KernelFunctions.Wendland2), domain,
                              adjacency=adjacency, referenceParticles=r,
                              queryTangentState=zero, referenceTangentState=ref)
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
@pytest.mark.parametrize('direction', list(OperationDirection))
@pytest.mark.parametrize('crk', [False, True])
def test_rebuilt_corrections_match_masked_primal_chain(device, traversal, direction, crk):
    q, r, domain, _ = cloud(device)
    # Five noncollinear reference particles per category. Cross-direction
    # volume policies can still select a fallback; parity covers that existing
    # forward policy as well as full-rank and rejected query rows.
    points = r.positions
    r = replace(r, positions=torch.cat([points + .01 * k for k in range(4)]),
                supports=r.supports.repeat(4), masses=r.masses.repeat(4),
                densities=r.densities.repeat(4),
                kinds=torch.arange(4, dtype=torch.int32, device=device).repeat_interleave(5))
    adjacency = radiusSearchCompactHashMap(q, domain, referenceParticles=r,
                                          mode=SupportScheme.SuperSymmetric)
    adj = {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient,
                                operationMode=direction, gradientMode=GradientScheme.Difference)
    inputs = tuple(t.detach().requires_grad_() for t in (q.positions, r.positions, q.supports, r.supports))
    tangents = tuple(.04 * (torch.arange(t.numel(), dtype=t.dtype, device=device).reshape_as(t) * 1.7 + k).sin()
                     for k, t in enumerate(inputs))
    tq = ParticleTangentState(tangents[0], tangents[2], None)
    tr = ParticleTangentState(tangents[1], tangents[3], None)
    fq, fr = q.positions[:, 0].sin(), r.positions[:, 1].cos()

    def evaluate(args, with_jvp=False):
        query = replace(q, positions=args[0], supports=args[2])
        ref = replace(r, positions=args[1], supports=args[3])
        state, dstate, renorm, drenorm = None, None, None, None
        if crk:
            if with_jvp:
                _, _, state, dstate = computeCRKFactorsJVP(
                    query, domain, KernelFunctions.Wendland2, tq, operationMode=direction,
                    adjacency=adj, referenceState=ref, referenceTangentState=tr)
            else:
                _, _, state = computeCRKFactors(query, domain, KernelFunctions.Wendland2,
                    operationMode=direction, adjacency=adj, referenceState=ref)
        if with_jvp:
            renorm, drenorm = computeRenormalizationMatricesJVP(
                query, props, domain, tq, referenceTangentState=tr,
                referenceParticles=ref, adjacency=adj, returnEigVals=False)
        else:
            renorm = computeRenormalizationMatrices(query, props, domain,
                referenceParticles=ref, adjacency=adj, returnEigVals=False)
        kwargs = dict(referenceParticles=ref, adjacency=adj, queryValues=fq, referenceValues=fr,
                      crkState=state, renormalizationState=renorm)
        if not with_jvp:
            return warpOperation(query, props, domain, **kwargs)
        return warpOperationJVP(query, props, domain, **kwargs,
                                queryTangentState=tq, referenceTangentState=tr,
                                crkTangentState=dstate, renormalizationTangentState=drenorm)

    output = evaluate(inputs)
    cotangent = torch.arange(output.numel(), dtype=output.dtype, device=device).reshape_as(output).sin()
    gradient = torch.autograd.grad((output * cotangent).sum(), inputs)
    actual = evaluate(inputs, with_jvp=True)
    assert torch.isfinite(actual).all()
    expected = sum((g * v).sum() for g, v in zip(gradient, tangents))
    tol = 3e-4 if output.dtype == torch.float32 else 3e-9
    torch.testing.assert_close((actual * cotangent).sum(), expected, rtol=tol, atol=tol)
    eps = .02 if output.dtype == torch.float32 else 1e-5
    fd = (evaluate(tuple(x + eps * v for x, v in zip(inputs, tangents))) -
          evaluate(tuple(x - eps * v for x, v in zip(inputs, tangents)))) / (2 * eps)
    torch.testing.assert_close(actual, fd, rtol=3e-3 if output.dtype == torch.float32 else 3e-7,
                               atol=5e-5 if output.dtype == torch.float32 else 3e-8)


@pytest.mark.parametrize('operation', [WarpOperation.Density, WarpOperation.Gradient])
@pytest.mark.parametrize('direction', list(OperationDirection))
@pytest.mark.parametrize('traversal', ['csr', 'grid'])
def test_torch_dual_uses_frozen_interaction_rules(device, operation, direction, traversal):
    from torch.autograd import forward_ad
    q, r, domain, adjacency = cloud(device)
    adj = adjacency if traversal == 'csr' else adjacency.hashMap
    qp, rp = q.positions.requires_grad_(), r.positions.requires_grad_()
    dq, dr = qp.detach().cos() * .05, rp.detach().sin() * .07
    props = OperationProperties(operation=operation, kernel=KernelFunctions.Wendland2,
                                operationMode=direction)
    kwargs = dict(referenceValues=r.positions[:, 1].cos().detach(),
                  queryValues=q.positions[:, 0].sin().detach()) if operation == WarpOperation.Gradient else {}
    output = warpOperation(q, props, domain, referenceParticles=r, adjacency=adj, **kwargs)
    vq, vr = torch.autograd.grad(output.sum(), (qp, rp))
    with forward_ad.dual_level():
        query = replace(q, positions=forward_ad.make_dual(qp.detach(), dq))
        ref = replace(r, positions=forward_ad.make_dual(rp.detach(), dr))
        result = warpOperation(query, props, domain, referenceParticles=ref, adjacency=adj, **kwargs)
        primal, tangent = forward_ad.unpack_dual(result)
        tol = 2e-4 if qp.dtype == torch.float32 else 2e-9
        torch.testing.assert_close(primal, output, rtol=0, atol=0)
        torch.testing.assert_close(tangent.sum(), (vq * dq).sum() + (vr * dr).sum(), rtol=tol, atol=tol)


@pytest.mark.parametrize('operation', [WarpOperation.Density, WarpOperation.Gradient])
@pytest.mark.parametrize('direction', [OperationDirection.AllToAll, OperationDirection.FluidToBoundary,
                                       OperationDirection.GhostToFluid])
@pytest.mark.parametrize('traversal', ['csr', 'grid'])
def test_mixed_kind_reverse_over_jvp_gradcheck(device, operation, direction, traversal):
    if get_torch_precision() != torch.float64:
        pytest.skip('gradcheck requires a float64 process')
    q, r, domain, adjacency = cloud(device)
    adj = adjacency if traversal == 'csr' else adjacency.hashMap
    props = OperationProperties(operation=operation, kernel=KernelFunctions.Wendland2,
                                operationMode=direction)
    values = dict(queryValues=q.positions[:, 0].sin(), referenceValues=r.positions[:, 1].cos())
    if operation == WarpOperation.Density:
        values = {}

    def f(qp, rp, dq, dr):
        return warpOperationJVP(replace(q, positions=qp), props, domain, adjacency=adj,
            referenceParticles=replace(r, positions=rp), **values,
            queryTangentState=ParticleTangentState(dq, None, None),
            referenceTangentState=ParticleTangentState(dr, None, None))

    inputs = tuple(x.detach().requires_grad_() for x in
                   (q.positions, r.positions, q.positions.cos() * .02, r.positions.sin() * .03))
    assert torch.autograd.gradcheck(f, inputs)


@pytest.mark.parametrize('direction', list(OperationDirection))
def test_density_hvp_filters_match_primal_gradient_differences(device, direction):
    p, _, domain, _ = cloud(device)
    adjacency = radiusSearchCompactHashMap(p, domain)
    i, j = adjacency.livePairs()
    # The existing HVP contract excludes equal-ID pairs. Use the same fixed
    # pair set in the independent forward/reverse finite-difference oracle.
    keep = i != j
    i, j = i[keep], j[keep]
    counts = torch.bincount(i.long(), minlength=4).to(torch.int32)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1])).to(get_torch_offset_dtype())
    without_self = AdjacencyList(i, j, counts, offsets, 4, 4, numEdges=len(i))
    props = OperationProperties(operation=WarpOperation.Density, kernel=KernelFunctions.Wendland2,
                                operationMode=direction)
    dq, dr = p.positions.cos() * .07, p.positions.sin() * .11
    actual = warpOperationHVP(p, props, domain, dq, adjacency=adjacency, tangentReferencePositions=dr)

    def gradient(step):
        pos = (p.positions + step * dq).detach().requires_grad_()
        ref = replace(p, positions=p.positions + step * dr)
        result = warpOperation(replace(p, positions=pos), props, domain,
                               referenceParticles=ref, adjacency=without_self)
        return torch.autograd.grad(result.sum(), pos)[0]

    eps = .002 if p.positions.dtype == torch.float32 else 1e-5
    expected = (gradient(eps) - gradient(-eps)) / (2 * eps)
    torch.testing.assert_close(actual, expected, rtol=3e-3 if p.positions.dtype == torch.float32 else 2e-5,
                               atol=3e-4 if p.positions.dtype == torch.float32 else 3e-6)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_crk_constant_fallback_has_finite_zero_derivatives(device, dim):
    # The first two rows are low-count; the third has enough neighbors but a
    # singular moment matrix. No derivative may leak through their fallback.
    dtype = get_torch_precision()
    shapes = [(4,), (4, dim), (4, dim, dim), (4, dim), (4, dim, dim), (4, dim, dim, dim)]
    inputs = tuple(torch.zeros(shape, dtype=dtype, device=device, requires_grad=True) for shape in shapes)
    with torch.no_grad():
        if dim > 1:
            v = torch.arange(1, dim + 1, dtype=dtype, device=device)
            inputs[2][3] = v[:, None] * v[None]
        inputs[0][3] = 1
    counts = torch.tensor([0, 1, 7, 7], dtype=torch.int32, device=device)
    supports = inputs[0].new_ones(4)

    def f(*args):
        return computeCRKTermsWarp(*args, counts, supports)

    output, jvp = torch.autograd.functional.jvp(f, inputs, tuple(torch.ones_like(t) for t in inputs), create_graph=True)
    for value in jvp:
        assert torch.isfinite(value).all() and torch.count_nonzero(value) == 0
    grads = torch.autograd.grad(sum(t.sum() for t in output), inputs)
    for value in grads:
        assert torch.isfinite(value).all() and torch.count_nonzero(value) == 0
    torch.testing.assert_close(output[0], torch.ones_like(output[0]), rtol=0, atol=0)
    for value in output[1:]:
        assert torch.count_nonzero(value) == 0


@pytest.mark.parametrize('role', ['query', 'reference'])
def test_mutating_frozen_kinds_before_backward_is_detected(device, role):
    q, r, domain, adjacency = cloud(device)
    q = replace(q, positions=q.positions.requires_grad_())
    result = warpOperationJVP(q, OperationProperties(operation=WarpOperation.Density,
        kernel=KernelFunctions.Wendland2), domain, adjacency=adjacency, referenceParticles=r,
        queryTangentState=ParticleTangentState(torch.ones_like(q.positions) * .1, None, None))
    state = q if role == 'query' else r
    state.kinds.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        result.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream/capture')
def test_kind_masks_follow_current_stream_and_captured_updates():
    q, r, domain, adjacency = cloud(torch.device('cuda'))
    props = OperationProperties(operation=WarpOperation.Density, kernel=KernelFunctions.Wendland2)
    zero = ParticleTangentState(torch.zeros_like(q.positions), None, None)
    ref = ParticleTangentState(torch.zeros_like(r.positions), None, torch.ones_like(r.masses))

    def jvp():
        return warpOperationJVP(q, props, domain, adjacency=adjacency, referenceParticles=r,
                                queryTangentState=zero, referenceTangentState=ref)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        r.kinds.fill_(2)
        for _ in range(3):
            expected_zero = jvp()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.count_nonzero(expected_zero) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = jvp()
    for kind in (0, 2, 1):
        r.kinds.fill_(kind)
        graph.replay()
        reference = jvp()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        assert bool((output == 0).all()) if kind == 2 else bool((output > 0).all())
