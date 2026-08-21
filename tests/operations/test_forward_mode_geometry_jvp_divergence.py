"""In-process standing test for `warpOperationJVP`'s geometry JVP Divergence
branch (`warpier_tier2_operators_plan.md` Step 5, `warpier_adjoint.md` Tier
2.2): asserts `computeSPHDivergenceGeometryJVP` matches a reverse-mode-
Jacobian reference on the production `warpOperation(Divergence)` call
(`divergenceDotMode=False`, the only mode in geometry JVP scope), same pattern as
`test_forward_mode_geometry_jvp_gradient.py`.
"""

from __future__ import annotations

import pytest
import torch

from warpSPHCore import (
    DomainDescription,
    OperationProperties,
    ParticleState,
    ParticleTangentState,
    radiusSearchCompactHashMap,
    warpOperation,
    warpOperationJVP,
)
from warpSPHCore.enumTypes import GradientScheme, OperationDirection, SupportScheme, WarpOperation, KernelFunctions

DEVICE = torch.device("cpu")
DTYPE = torch.float32
KERNEL = KernelFunctions.Wendland2


def _make_domain(dim: int, margin: float = 10.0) -> DomainDescription:
    return DomainDescription(
        min=torch.tensor([-margin] * dim, dtype=DTYPE, device=DEVICE),
        max=torch.tensor([margin] * dim, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False] * dim, device=DEVICE),
        dim=dim,
    )


def _grid_case_2d(n_per_side: int = 3, spacing: float = 0.4):
    coords = torch.linspace(-(n_per_side - 1) / 2 * spacing, (n_per_side - 1) / 2 * spacing,
                             n_per_side, dtype=DTYPE, device=DEVICE)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    n = positions.shape[0]
    h = max(2.5 * spacing, 1e-3)
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE) * (1.0 + 0.15 * torch.linspace(-1, 1, n, dtype=DTYPE))
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    return positions, supports, masses


def _densities_for(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    return warpOperation(p, props, domain, adjacency=adjacency).detach()


def _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme):
    n = positions.shape[0]
    dim = positions.shape[1]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(hash((mode, scheme)) % (2 ** 31))
    queryValues = torch.randn(n, dim, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dim, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Divergence,
                                supportMode=mode, operationMode=OperationDirection.AllToAll,
                                gradientMode=scheme, divergenceDotMode=False)

    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency, queryValues=queryValues, referenceValues=referenceValues)

    pos0 = positions.clone().requires_grad_(True)
    sup0 = supports.clone().requires_grad_(True)
    mass0 = masses.clone().requires_grad_(True)
    density0 = densities.clone().requires_grad_(True)

    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1
    dmass = torch.randn_like(masses)
    ddensity = torch.randn_like(densities) * 0.1

    J = torch.autograd.functional.jacobian(f, (pos0, sup0, mass0, density0), vectorize=False)
    out = f(pos0, sup0, mass0, density0).detach()
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, (dpos, dsup, dmass, ddensity)):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    reference = acc.reshape(out.shape)

    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    assembled = warpOperationJVP(
        p0, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddensity),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddensity),
        queryValues=queryValues, referenceValues=referenceValues,
    )
    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


def _check_combined_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme):
    # warpier_tier2_combined_jvp_plan.md: geometry tangent + value tangent
    # together should equal the geometry JVP and value JVP contributions summed.
    n = positions.shape[0]
    dim = positions.shape[1]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(hash(("combined", mode, scheme)) % (2 ** 31))
    queryValues = torch.randn(n, dim, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dim, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Divergence,
                                supportMode=mode, operationMode=OperationDirection.AllToAll,
                                gradientMode=scheme, divergenceDotMode=False)

    def f(pos, sup, mass, density, qval, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency, queryValues=qval, referenceValues=rval)

    pos0 = positions.clone().requires_grad_(True)
    sup0 = supports.clone().requires_grad_(True)
    mass0 = masses.clone().requires_grad_(True)
    density0 = densities.clone().requires_grad_(True)
    qval0 = queryValues.clone().requires_grad_(True)
    rval0 = referenceValues.clone().requires_grad_(True)

    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1
    dmass = torch.randn_like(masses)
    ddensity = torch.randn_like(densities) * 0.1
    dqval = torch.randn_like(queryValues)
    drval = torch.randn_like(referenceValues)

    J = torch.autograd.functional.jacobian(f, (pos0, sup0, mass0, density0, qval0, rval0), vectorize=False)
    out = f(pos0, sup0, mass0, density0, qval0, rval0).detach()
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, (dpos, dsup, dmass, ddensity, dqval, drval)):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    reference = acc.reshape(out.shape)

    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    assembled = warpOperationJVP(
        p0, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddensity),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddensity),
        tangentQueryValues=dqval, tangentReferenceValues=drval,
        queryValues=queryValues, referenceValues=referenceValues,
    )
    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("scheme", list(GradientScheme))
def test_divergenceGeometryJVP_combined_matches_jacobian_reference_2d(scheme):
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    _check_combined_jacobian_reference(positions, supports, masses, domain, adjacency, SupportScheme.Gather, scheme)


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric])
def test_divergenceGeometryJVP_matches_jacobian_reference_2d(mode, scheme):
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme)


def _minimal_case():
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Divergence,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                gradientMode=GradientScheme.Symmetric, divergenceDotMode=False)
    queryValues = torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    return positions, p0, domain, adjacency, props, queryValues, referenceValues


def test_divergenceGeometryJVP_rejects_missing_values():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    with pytest.raises(ValueError, match="queryValues"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None))


def test_divergenceGeometryJVP_rejects_dotMode():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    dotProps = OperationProperties(kernel=KERNEL, operation=WarpOperation.Divergence,
                                   supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                   gradientMode=GradientScheme.Symmetric, divergenceDotMode=True)
    with pytest.raises(NotImplementedError, match="geometry JVP"):
        warpOperationJVP(p0, dotProps, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None),
                         queryValues=qv, referenceValues=rv)


def test_divergenceGeometryJVP_rejects_consistentDivergence():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    with pytest.raises(NotImplementedError, match="geometry JVP"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None),
                         consistentDivergence=True,
                         queryValues=qv, referenceValues=rv)


def test_divergenceGeometryJVP_accepts_crkState():
    # warpier_tier2_correction_jvp_plan.md phase (e): CRK tangent extension
    # landed for Divergence (and Curl/Laplacian(Brookshaw)) -- crkState (with
    # or without crkTangentState) no longer raises here. See
    # test_gradientGeometryJVP_accepts_crkState in
    # test_forward_mode_geometry_jvp_gradient.py for phase (c)'s Gradient-only
    # precedent this extends.
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    from warpSPHCore.dataTypes import CRKState, CRKTangentState
    n = positions.shape[0]
    dummy = CRKState(A=torch.zeros(n, dtype=DTYPE), B=torch.zeros(n, 2, dtype=DTYPE),
                     gradA=torch.zeros(n, 2, dtype=DTYPE), gradB=torch.zeros(n, 2, 2, dtype=DTYPE))
    result = warpOperationJVP(p0, props, domain, adjacency=adjacency,
                              queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None),
                              crkState=dummy,
                              queryValues=qv, referenceValues=rv)
    assert torch.isfinite(result).all()

    dummyTangent = CRKTangentState(A=torch.zeros(n, dtype=DTYPE), B=torch.zeros(n, 2, dtype=DTYPE),
                                    gradA=torch.zeros(n, 2, dtype=DTYPE), gradB=torch.zeros(n, 2, 2, dtype=DTYPE))
    with pytest.raises(ValueError, match="crkTangentState requires crkState"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None),
                         crkTangentState=dummyTangent,
                         queryValues=qv, referenceValues=rv)


def test_divergenceGeometryJVP_geometryOnly_unchanged_when_combination_allowed():
    # Regression guard (warpier_tier2_combined_jvp_plan.md step 5): the
    # geometry-only path (no value tangent supplied) must return exactly
    # what it did before the combined path was added.
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    dpos = torch.zeros_like(positions)
    viaJVP = warpOperationJVP(p0, props, domain, adjacency=adjacency,
                              queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None), queryValues=qv, referenceValues=rv)
    from warpSPHCore.coreOperations import computeSPHDivergenceGeometryJVP
    direct = computeSPHDivergenceGeometryJVP(
        p0, domain, props.kernel, props.supportMode, adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None), queryValues=qv, referenceValues=rv,
        gradientMode=props.gradientMode,
    )
    torch.testing.assert_close(viaJVP, direct, rtol=0, atol=0)


def test_divergenceGeometryJVP_rejects_tangentQueryMasses():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    n = positions.shape[0]
    with pytest.raises(NotImplementedError, match="geometry JVP"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(
                             positions=torch.zeros_like(positions), supports=None,
                             masses=torch.zeros(n, dtype=DTYPE),
                         ),
                         queryValues=qv, referenceValues=rv)


def test_divergenceGeometryJVP_grid_traversal_matches_adjacency_traversal():
    # computeSPHDivergenceGeometryJVP (CSR, warpier_tier2_jvp_csr_backend_plan.md)
    # also supports grid (CompactHashMap) traversal -- exercised here via a
    # direct import against the low-level function, independent of
    # warpOperationJVP.
    from warpSPHCore.coreOperations import computeSPHDivergenceGeometryJVP

    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    dim = positions.shape[1]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    hashMap = adjacency.hashMap
    assert hashMap is not None

    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)

    torch.manual_seed(2)
    queryValues = torch.randn(n, dim, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dim, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1
    dmass = torch.randn_like(masses)
    ddensity = torch.randn_like(densities) * 0.1

    common = dict(
        queryParticles=p0, domain=domain, kernel=KERNEL, supportMode=SupportScheme.Gather,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None, densities=ddensity),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=dmass, densities=ddensity),
        queryValues=queryValues, referenceValues=referenceValues,
        gradientMode=GradientScheme.Symmetric,
    )
    viaAdjacency = computeSPHDivergenceGeometryJVP(adjacency=adjacency, **common)
    viaGrid = computeSPHDivergenceGeometryJVP(adjacency=hashMap, **common)

    torch.testing.assert_close(viaGrid, viaAdjacency, rtol=1e-5, atol=1e-6)
