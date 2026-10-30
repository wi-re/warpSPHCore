"""Row identity, correction roles, prepared lifetimes and flat numerical parity."""
from dataclasses import replace

import pytest
import torch
import torch.autograd.forward_ad as fwAD

from warpSPHCore import (DomainDescription, ParticleState, OperationProperties, KernelFunctions,
                         SupportScheme, WarpOperation, GradientScheme, LaplacianScheme,
                         radiusSearchCompactHashMap, warpOperation, OperationDirection)
from warpSPHCore.autograd import (Corrections, SPHContext, PreparedLauncher, ExtraKind, ExtraSpec,
                                  RowRole, launchOperator, setNeighborLanes)
from warpSPHCore.coreOperations.wp_density import _DENSITY_SPEC
from warpSPHCore.coreOperations.wp_gradient import _GRADIENT_SPEC
from warpSPHCore.coreOperations.wp_covariance import _COVARIANCE_SPEC
from warpSPHCore.crk import computeCRKFactors
from warpSPHCore.dataTypes import CRKState, GradHState, RenormalizationState
from warpSPHCore.radiusSearch import (radiusSearchRowChunks, RowChunkPolicy,
                                     buildChunkedVerletList, deferVerletChecks)
from warpSPHCore.renorm import computeRenormalizationMatrices_
from warpSPHCore.type_config import get_torch_precision


@pytest.fixture(autouse=True)
def lanes(request):
    setNeighborLanes(1)
    yield
    setNeighborLanes(None)


def sample(device, dim=2, populations='self', sizes=None):
    dtype = get_torch_precision()
    def state(n, shift):
        t = torch.arange(n, dtype=dtype, device=device)
        x = torch.stack([torch.sin(t * (k + .7) + shift) * .4 for k in range(dim)], 1)
        return ParticleState(x, 1.1 + .2 * torch.cos(t), .8 + .03 * t,
                             (torch.arange(n, device=device) % 3 == 0).to(torch.int32), 1.4 + .02 * t)
    nq, nr = sizes or ((11, 19) if populations == 'unequal' else (19, 19))
    q = state(nq, .12)
    r = q if populations == 'self' and sizes is None else state(nr, .77)
    domain = DomainDescription(torch.full((dim,), -2., dtype=dtype, device=device),
                               torch.full((dim,), 2., dtype=dtype, device=device),
                               torch.zeros(dim, dtype=torch.bool, device=device), dim)
    policy = RowChunkPolicy(edgeBudget=80, maxRows=3)
    chunked = radiusSearchRowChunks(q, domain, referenceParticles=r, policy=policy,
                                    mode=SupportScheme.SuperSymmetric)
    flat = radiusSearchCompactHashMap(q, domain, referenceParticles=r, mode=SupportScheme.SuperSymmetric)
    return q, r, domain, policy, chunked, flat


def correction(q, r, kind):
    def terms(p):
        n, dim = p.positions.shape
        A = p.masses * .6
        return CRKState(A, p.positions * .13, p.positions * -.07,
                        torch.eye(dim, device=A.device, dtype=A.dtype).repeat(n, 1, 1) * A[:, None, None] * .04)
    if kind == 'none':
        return {}
    vol = dict(queryVolumes=q.masses / q.densities, referenceVolumes=r.masses / r.densities)
    if kind == 'crk':
        return dict(**vol, crkState=(terms(q), terms(r)))
    matrices = torch.eye(q.positions.shape[1], device=q.positions.device, dtype=q.positions.dtype)[None]
    return dict(**vol, gradHState=GradHState(q.densities * .73, r.densities * .81),
                renormalizationState=RenormalizationState(matrices * q.masses[:, None, None]))


def equal(a, b):
    a, b = ((a,), (b,)) if isinstance(a, torch.Tensor) else (a, b)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('populations', ['self', 'equal', 'unequal'])
@pytest.mark.parametrize('corrections', ['none', 'crk', 'gradH-renorm'])
def test_shared_operators_keep_query_and_reference_identity(device, dim, populations, corrections):
    q, r, domain, _, chunked, flat = sample(device, dim, populations)
    for operation in (WarpOperation.Density, WarpOperation.Covariance, WarpOperation.Interpolate,
                      WarpOperation.Gradient, WarpOperation.Divergence, WarpOperation.Curl, WarpOperation.Laplacian):
        props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=operation,
                                    supportMode=SupportScheme.SuperSymmetric,
                                    gradientMode=GradientScheme.Symmetric, laplacianMode=LaplacianScheme.Brookshaw)
        values = dict(queryValues=q.positions.square() + .1, referenceValues=r.positions.sin() - .2)
        kwargs = dict(referenceParticles=r, **values, **correction(q, r, corrections),
                      covarianceReturnNumNeighbors=True)
        # Interpolation's public normalization also handles role-paired CRK states.
        expected = warpOperation(q, props, domain, adjacency=flat, **kwargs)
        actual = warpOperation(q, props, domain, adjacency=chunked, **kwargs)
        equal(actual, expected)
    assert len(chunked.chunks) > 1


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('populations', ['self', 'unequal'])
def test_correction_construction_pipeline(device, dim, populations):
    q, r, domain, _, chunks, flat = sample(device, dim, populations)
    def run(adj):
        volume, density, crk = computeCRKFactors(q, domain, KernelFunctions.Wendland2,
                                               referenceState=r, adjacency=adj)
        props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Gradient)
        C, eigenvalues, renorm = computeRenormalizationMatrices_(q, props, domain, adjacency=adj, referenceParticles=r)
        return (volume, density, *vars(crk).values(), C, eigenvalues, renorm)
    equal(run(chunks), run(flat))


@pytest.mark.parametrize('sizes', [(0, 0), (0, 5), (3, 0)])
def test_empty_particle_outputs(device, sizes):
    q, r, domain, _, chunks, flat = sample(device, sizes=sizes)
    for spec in (_DENSITY_SPEC, _COVARIANCE_SPEC):
        ctx = SPHContext(q, OperationProperties(kernel=KernelFunctions.Wendland2), domain, chunks, r)
        expected = launchOperator(spec, replace(ctx, adjacency=flat))
        equal(launchOperator(spec, ctx), expected)
        equal(PreparedLauncher(ctx).launch(spec), expected)


def test_prepared_queries_alias_full_references_before_slicing(device):
    q, _, domain, _, chunks, flat = sample(device)
    dim = q.positions.shape[1]
    c = Corrections(volumes=(q.masses / q.densities, None),
                    crk=correction(q, q, 'crk')['crkState'][0], gradH=GradHState(q.densities))
    ctx = SPHContext(q, OperationProperties(kernel=KernelFunctions.Wendland2), domain, chunks, corrections=c)
    prepared = PreparedLauncher(ctx)
    # Unequal role values even though both populations alias the same particle state.
    extras = dict(numDims=0, flatInputShape=1, flatOutputShape=dim,
                  queryValuesFlat=q.masses[:, None], referenceValuesFlat=q.densities[:, None])
    for _ in range(2):
        equal(prepared.launch(_GRADIENT_SPEC, **extras),
              launchOperator(_GRADIENT_SPEC, replace(ctx, adjacency=flat), **extras))
        equal(launchOperator(_GRADIENT_SPEC, ctx, **extras), prepared.launch(_GRADIENT_SPEC, **extras))
        extras['queryValuesFlat'] = extras['queryValuesFlat'] * 1.03
        extras['referenceValuesFlat'] = extras['referenceValuesFlat'] + .04


def test_old_prepared_storage_is_rejected_only_when_reused_in_place(device):
    q, _, domain, policy, _, _ = sample(device)
    old = buildChunkedVerletList(q, domain, policy=policy, verletScale=1.1)
    ctx = SPHContext(q, OperationProperties(kernel=KernelFunctions.Wendland2), domain, old)
    prepared = PreparedLauncher(ctx)
    expected = prepared.launch(_DENSITY_SPEC)
    fresh = buildChunkedVerletList(q, domain, policy=policy, verletScale=1.1,
                                   priorNeighborhood=old, forceRebuild=True)
    equal(prepared.launch(_DENSITY_SPEC), expected)
    fresh = buildChunkedVerletList(q, domain, policy=policy, verletScale=1.1,
                                   priorNeighborhood=old, forceRebuild=True, inPlace=True)
    for launch in (lambda: prepared.launch(_DENSITY_SPEC), lambda: launchOperator(_DENSITY_SPEC, ctx),
                   lambda: PreparedLauncher(ctx),
                   lambda: buildChunkedVerletList(q, domain, policy=policy, priorNeighborhood=old)):
        with pytest.raises(ValueError, match='overwritten'):
            launch()
    equal(PreparedLauncher(replace(ctx, adjacency=fresh)).launch(_DENSITY_SPEC), expected)
    ctx.adjacency = fresh
    with pytest.raises(ValueError, match='new prepared'):
        prepared.launch(_DENSITY_SPEC)


def test_rejects_unsupported_specs_roles_ad_and_deferred_contexts(device):
    q, r, domain, policy, chunks, _ = sample(device)
    ctx = SPHContext(q, OperationProperties(kernel=KernelFunctions.Wendland2), domain, chunks, r)
    with pytest.raises(NotImplementedError, match='opted'):
        launchOperator(replace(_DENSITY_SPEC, rowChunked=False), ctx)
    unknown = replace(_DENSITY_SPEC, extras=(ExtraSpec('mystery', ExtraKind.TENSOR),))
    with pytest.raises(NotImplementedError, match='explicit row role'):
        launchOperator(unknown, ctx, mystery=q.masses)
    for role, tensor in ((RowRole.QUERY, q.masses[:2]), (RowRole.REFERENCE, r.masses[:2])):
        spec = replace(unknown, extras=(ExtraSpec('mystery', ExtraKind.TENSOR, role),))
        with pytest.raises(ValueError, match='particle rows'):
            launchOperator(spec, ctx, mystery=tensor)
    with pytest.raises(NotImplementedError, match='AD'):
        launchOperator(_DENSITY_SPEC, replace(ctx, query=replace(q, masses=q.masses.clone().requires_grad_())))
    with fwAD.dual_level():
        dual = fwAD.make_dual(q.positions, torch.ones_like(q.positions))
        with pytest.raises(NotImplementedError, match='AD'):
            PreparedLauncher(replace(ctx, query=replace(q, positions=dual)))
    with deferVerletChecks():
        with pytest.raises(NotImplementedError, match='graph'):
            launchOperator(_DENSITY_SPEC, ctx)
    owners = radiusSearchRowChunks(q, domain, policy=policy, numOwners=2)
    with pytest.raises(NotImplementedError, match='single-owner'):
        launchOperator(_DENSITY_SPEC, replace(ctx, adjacency=owners))


def test_nondefault_cuda_stream_and_existing_neighbor_lanes(device):
    if device.type != 'cuda':
        pytest.skip('CUDA stream and tiled lanes')
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        q, r, domain, _, chunks, flat = sample(device, 3, 'unequal')
        for lanes in (1, 32):
            setNeighborLanes(lanes)
            ctx = SPHContext(q, OperationProperties(kernel=KernelFunctions.Wendland2), domain, chunks, r)
            for spec in (_DENSITY_SPEC, _COVARIANCE_SPEC):
                expected = launchOperator(spec, replace(ctx, adjacency=flat))
                equal(launchOperator(spec, ctx), expected)
                equal(PreparedLauncher(ctx).launch(spec), expected)
    stream.synchronize()


@pytest.mark.parametrize('mode', list(SupportScheme))
@pytest.mark.parametrize('direction', [OperationDirection.FluidToBoundary, OperationDirection.BoundaryToFluid,
                                     OperationDirection.FluidToFluid, OperationDirection.TrueAllToToAll])
def test_direction_masks_and_support_modes_with_periodic_rows(device, mode, direction):
    q, r, domain, policy, _, _ = sample(device, populations='unequal')
    domain = replace(domain, min=domain.min * .25, max=domain.max * .25,
                     periodic=torch.ones_like(domain.periodic))
    chunks = radiusSearchRowChunks(q, domain, referenceParticles=r, policy=policy, mode=mode)
    flat = radiusSearchCompactHashMap(q, domain, referenceParticles=r, mode=mode)
    for op in (WarpOperation.Density, WarpOperation.Gradient, WarpOperation.Covariance):
        props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=op,
                                    supportMode=mode, operationMode=direction)
        kwargs = dict(referenceParticles=r, queryValues=q.masses.square(), referenceValues=r.densities.sin())
        equal(warpOperation(q, props, domain, adjacency=chunks, **kwargs),
              warpOperation(q, props, domain, adjacency=flat, **kwargs))


def test_conditional_tensor_roles_and_shape_guards_before_allocation(device, monkeypatch):
    q, r, domain, _, chunks, _ = sample(device, populations='unequal')
    ctx = SPHContext(q, OperationProperties(kernel=KernelFunctions.Wendland2), domain, chunks, r)
    # A declaration must follow the actual kernel branch, never infer identity from shape.
    spec = replace(_DENSITY_SPEC, extras=(ExtraSpec('enabled', ExtraKind.SCALAR),
        ExtraSpec('optional', ExtraKind.TENSOR,
                  lambda ctx, e: RowRole.QUERY if e['enabled'] else RowRole.REPLICATED)))
    from warpSPHCore.autograd.row_dispatch import _validateSpec
    assert _validateSpec(spec, ctx, dict(enabled=False, optional=q.masses[:1]))['optional'] is RowRole.REPLICATED
    with pytest.raises(ValueError, match='particle rows'):
        _validateSpec(spec, ctx, dict(enabled=True, optional=q.masses[:1]))
    assert _validateSpec(spec, ctx, dict(enabled=True, optional=q.masses))['optional'] is RowRole.QUERY
    # No kernel/allocation follows invalid row metadata or an unsafe CRK alias.
    module = __import__('warpSPHCore.autograd.row_dispatch', fromlist=['_allocate_output'])
    monkeypatch.setattr(module, '_allocate_output', lambda *a, **k: pytest.fail('allocated before validation'))
    with pytest.raises(ValueError, match='particle rows'):
        launchOperator(_DENSITY_SPEC, replace(ctx, query=replace(q, masses=q.masses[:2])))
    c = Corrections(crk=correction(q, q, 'crk')['crkState'][0])
    with pytest.raises(ValueError, match='reference CRK'):
        launchOperator(_COVARIANCE_SPEC, replace(ctx, corrections=c))
