"""Complete correction/AD chains and an SPH-to-constrained-solver consumer."""

from dataclasses import replace

import pytest
import torch
import warp as wp

from warpSPHCore import (
    DomainDescription, GradientScheme, KernelFunctions, LaplacianScheme,
    OperationProperties, ParticleState, ParticleTangentState, SupportScheme,
    WarpOperation, radiusSearchCompactHashMap, warpOperation, warpOperationJVP,
)
from warpSPHCore.crk import computeCRKFactors, computeCRKFactorsJVP
from warpSPHCore.linalg import LinearOperator, RestrictedSystem, gmres
from warpSPHCore.renorm import computeRenormalizationMatrices, computeRenormalizationMatricesJVP
from warpSPHCore.type_config import get_torch_precision


def case(device, traversal):
    dtype = get_torch_precision()
    axis = torch.linspace(-.3, .3, 3, dtype=dtype, device=device)
    xy = torch.cartesian_prod(axis, axis)
    h = torch.linspace(1.1, 1.3, len(xy), dtype=dtype, device=device)
    p = ParticleState(positions=xy, supports=h, masses=torch.full_like(h, .09),
                      kinds=torch.zeros(len(xy), dtype=torch.int32, device=device))
    domain = DomainDescription(min=xy.new_full((2,), -2), max=xy.new_full((2,), 2),
                               periodic=torch.zeros(2, dtype=torch.bool, device=device), dim=2)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=SupportScheme.SuperSymmetric)
    return p, domain, {'csr': adjacency, 'grid': adjacency.hashMap, 'implicit': None}[traversal]


def properties(operation, **kwargs):
    return OperationProperties(kernel=KernelFunctions.Wendland2, operation=operation,
                                supportMode=SupportScheme.Gather,
                                gradientMode=GradientScheme.Difference, **kwargs)


@pytest.mark.parametrize('traversal', ['csr', 'grid', 'implicit'])
@pytest.mark.parametrize('crk', [False, True])
def test_density_correction_gradient_chain_dot_and_finite_difference(device, traversal, crk):
    p, domain, adj = case(device, traversal)
    dp = p.positions.sin() * .1 + .03
    dh = p.supports.cos() * .02
    tangent = ParticleTangentState(positions=dp, supports=dh, masses=None)

    def evaluate(pos, support, with_jvp=False):
        state = replace(p, positions=pos, supports=support)
        density_props = properties(WarpOperation.Density)
        density = warpOperation(state, density_props, domain, adjacency=adj)
        if with_jvp:
            ddensity = warpOperationJVP(
                state, density_props, domain, adjacency=adj,
                queryTangentState=tangent, referenceTangentState=tangent)
        state = replace(state, densities=density)
        volume, dvolume, correction, dcorrection = None, None, None, None
        if crk:
            if with_jvp:
                volume, dvolume, correction, dcorrection = computeCRKFactorsJVP(
                    state, domain, KernelFunctions.Wendland2, tangent,
                    referenceTangentState=tangent, adjacency=adj)
            else:
                volume, _, correction = computeCRKFactors(state, domain, KernelFunctions.Wendland2, adjacency=adj)
        props = properties(WarpOperation.Gradient)
        if with_jvp:
            state_tangent = replace(tangent, densities=ddensity)
            renorm, drenorm = computeRenormalizationMatricesJVP(
                state, props, domain, state_tangent, referenceTangentState=state_tangent,
                referenceVolumes=volume, tangentReferenceVolumes=dvolume,
                adjacency=adj, returnEigVals=False)
        else:
            renorm = computeRenormalizationMatrices(
                state, props, domain, referenceVolumes=volume, adjacency=adj, returnEigVals=False)
        field = 1 + pos[:, 0].square() + .3 * pos[:, 1].sin()
        kwargs = dict(adjacency=adj, queryValues=field, referenceValues=field,
                      referenceVolumes=volume, crkState=correction, renormalizationState=renorm)
        if not with_jvp:
            return warpOperation(state, props, domain, **kwargs)
        dfield = 2 * pos[:, 0] * dp[:, 0] + .3 * pos[:, 1].cos() * dp[:, 1]
        return warpOperationJVP(
            state, props, domain, **kwargs,
            queryTangentState=state_tangent, referenceTangentState=state_tangent,
            tangentQueryValues=dfield, tangentReferenceValues=dfield,
            tangentReferenceVolumes=dvolume, crkTangentState=dcorrection,
            renormalizationTangentState=drenorm)

    pos, h = p.positions.detach().requires_grad_(), p.supports.detach().requires_grad_()
    assembled = evaluate(pos, h, with_jvp=True)
    output = evaluate(pos, h)
    cotangent = torch.arange(output.numel(), dtype=pos.dtype, device=device).reshape_as(output).sin()
    vpos, vh = torch.autograd.grad((output * cotangent).sum(), (pos, h))
    tol = 5e-4 if pos.dtype == torch.float32 else 2e-9
    torch.testing.assert_close((assembled * cotangent).sum(), (vpos * dp).sum() + (vh * dh).sum(),
                               rtol=tol, atol=tol)
    eps = .002 if pos.dtype == torch.float32 else 1e-5
    fd = (evaluate(pos + eps * dp, h + eps * dh) - evaluate(pos - eps * dp, h - eps * dh)) / (2 * eps)
    torch.testing.assert_close(assembled, fd, rtol=4e-3 if pos.dtype == torch.float32 else 2e-7,
                               atol=3e-4 if pos.dtype == torch.float32 else 2e-8)


@pytest.mark.parametrize('traversal', ['csr', 'grid'])
@pytest.mark.parametrize('crk', [False, True])
def test_corrected_sph_dirichlet_solve(device, traversal, crk):
    p, domain, adj = case(device, traversal)
    p.densities = warpOperation(p, properties(WarpOperation.Density), domain, adjacency=adj)
    volume, correction = None, None
    if crk:
        volume, _, correction = computeCRKFactors(p, domain, KernelFunctions.Wendland2, adjacency=adj)
    props = properties(WarpOperation.Laplacian, laplacianMode=LaplacianScheme.Brookshaw)
    renorm = computeRenormalizationMatrices(
        p, props, domain, referenceVolumes=volume, adjacency=adj, returnEigVals=False)

    def action(v):
        return -warpOperation(p, props, domain, queryValues=v, referenceValues=v,
                               referenceVolumes=volume, crkState=correction,
                               renormalizationState=renorm, adjacency=adj)

    # The correction need not produce a symmetric matrix. Declare only the
    # actual action and use GMRES, rather than inferring an SPD contract.
    op = LinearOperator((9, 9), p.positions.dtype, p.positions.device, action)
    free = torch.tensor([1, 3, 4, 5, 7], device=device)
    system = RestrictedSystem(op, free)
    exact = 1 + p.positions[:, 0].square() - .4 * p.positions[:, 1]
    boundary = exact.clone()
    boundary[free] = 0
    b = action(exact)
    rhs = system.liftRhs(b, boundary)
    tol = 3e-5 if b.dtype == torch.float32 else 2e-11
    result = gmres(system, rhs, restart=5, maxiter=30, rtol=tol, checkEvery=1)
    assert result.statusInt() >= 0
    solution = system.reconstruct(result.x, boundary)
    torch.testing.assert_close(solution, exact, rtol=tol * 3, atol=tol * 3)
    assert system.restrict(b - action(solution)).norm() <= tol * rhs.norm() * 2
    torch.testing.assert_close(solution[system.constrainedIndices],
                               boundary[system.constrainedIndices], rtol=0, atol=0)
    eye = torch.eye(9, dtype=b.dtype, device=device)
    dense = torch.stack([action(v) for v in eye], dim=1)
    expected = torch.linalg.solve(dense[free[:, None], free].double(), rhs.double())
    torch.testing.assert_close(solution[free].double(), expected, rtol=tol * 3, atol=tol * 3)
