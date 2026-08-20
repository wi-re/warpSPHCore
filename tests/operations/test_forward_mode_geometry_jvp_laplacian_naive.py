"""In-process standing test for `warpOperationJVP`'s geometry JVP Laplacian
(Naive scheme) branch (`warpier_tier2_operators_plan.md` Step 8,
`warpier_adjoint.md` Tier 2.3): asserts `computeSPHLaplacianNaiveGeometryJVP`
matches a reverse-mode-Jacobian reference on the production
`warpOperation(Laplacian, laplacianMode=Naive)` call, same pattern as
`test_forward_mode_geometry_jvp_laplacian_brookshaw.py`.

Also checks the genuine structural finding `warpier_adjoint.md` Tier 2.3
surfaced: unlike Tier 2.1/2.2's kernel value/gradient, `sphKernelLaplacian`
never gained an explicit `KernelMeanSymmetric` branch, so KernelMeanSymmetric
and SuperSymmetric give genuinely *different* results here (not identical,
as they are for every other operator in this plan).
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
from warpSPHCore.enumTypes import GradientScheme, LaplacianScheme, OperationDirection, SupportScheme, WarpOperation, KernelFunctions

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


def _line_case(n: int = 7, xmin: float = -1.0, xmax: float = 1.0):
    positions = torch.linspace(xmin, xmax, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    spacing = (xmax - xmin) / max(n - 1, 1)
    h = max(2.5 * spacing, 1e-3)
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE)
    supports = supports * (1.0 + 0.15 * torch.linspace(-1, 1, n, dtype=DTYPE))
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    return positions, supports, masses


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
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(hash((mode, scheme)) % (2 ** 31))
    queryValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                supportMode=mode, operationMode=OperationDirection.AllToAll,
                                gradientMode=scheme, laplacianMode=LaplacianScheme.Naive)

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
    return assembled


def _check_combined_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme):
    # warpier_tier2_combined_jvp_plan.md: geometry tangent + value tangent
    # together should equal the geometry JVP and value JVP contributions summed.
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(hash(("combined", mode, scheme)) % (2 ** 31))
    queryValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                supportMode=mode, operationMode=OperationDirection.AllToAll,
                                gradientMode=scheme, laplacianMode=LaplacianScheme.Naive)

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
def test_laplacianNaiveGeometryJVP_combined_matches_jacobian_reference_2d(scheme):
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    _check_combined_jacobian_reference(positions, supports, masses, domain, adjacency, SupportScheme.Gather, scheme)


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.SuperSymmetric])
def test_laplacianNaiveGeometryJVP_matches_jacobian_reference_1d(mode, scheme):
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.KernelMeanSymmetric)
    _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme)


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric])
def test_laplacianNaiveGeometryJVP_matches_jacobian_reference_2d(mode, scheme):
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme)


def test_laplacianNaiveGeometryJVP_kernelMeanSymmetric_differs_from_superSymmetric():
    # warpier_adjoint.md Tier 2.3's structural finding: sphKernelLaplacian never got
    # an explicit KernelMeanSymmetric branch, so (unlike every other operator in this
    # plan) the two schemes are NOT identical here -- the mirror image of Tier 2.2's
    # "assert identical" check.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.KernelMeanSymmetric)
    kms = _check_jacobian_reference(positions, supports, masses, domain, adjacency, SupportScheme.KernelMeanSymmetric, GradientScheme.Naive)
    ss = _check_jacobian_reference(positions, supports, masses, domain, adjacency, SupportScheme.SuperSymmetric, GradientScheme.Naive)
    assert not torch.allclose(kms, ss, atol=1e-6), "KernelMeanSymmetric and SuperSymmetric should genuinely differ for the Naive Laplacian JVP"


def _minimal_case():
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                gradientMode=GradientScheme.Symmetric, laplacianMode=LaplacianScheme.Naive)
    queryValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    return positions, p0, domain, adjacency, props, queryValues, referenceValues


def test_laplacianNaiveGeometryJVP_rejects_missing_values():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    with pytest.raises(ValueError, match="queryValues"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None))


def test_laplacianNaiveGeometryJVP_geometryOnly_unchanged_when_combination_allowed():
    # Regression guard (warpier_tier2_combined_jvp_plan.md step 5): the
    # geometry-only path (no value tangent supplied) must return exactly
    # what it did before the combined path was added.
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    dpos = torch.zeros_like(positions)
    viaJVP = warpOperationJVP(p0, props, domain, adjacency=adjacency,
                              queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None), queryValues=qv, referenceValues=rv)
    from warpSPHCore.coreOperations import computeSPHLaplacianNaiveGeometryJVP
    direct = computeSPHLaplacianNaiveGeometryJVP(
        p0, domain, props.kernel, props.supportMode, adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None), queryValues=qv, referenceValues=rv,
        gradientMode=props.gradientMode,
    )
    torch.testing.assert_close(viaJVP, direct, rtol=0, atol=0)


def test_laplacianNaiveGeometryJVP_grid_traversal_matches_adjacency_traversal():
    # computeSPHLaplacianNaiveGeometryJVP (CSR, warpier_tier2_jvp_csr_backend_plan.md)
    # also supports grid (CompactHashMap) traversal -- exercised here via a
    # direct import against the low-level function, independent of
    # warpOperationJVP.
    from warpSPHCore.coreOperations import computeSPHLaplacianNaiveGeometryJVP

    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    hashMap = adjacency.hashMap
    assert hashMap is not None

    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)

    torch.manual_seed(2)
    queryValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
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
    viaAdjacency = computeSPHLaplacianNaiveGeometryJVP(adjacency=adjacency, **common)
    viaGrid = computeSPHLaplacianNaiveGeometryJVP(adjacency=hashMap, **common)

    torch.testing.assert_close(viaGrid, viaAdjacency, rtol=1e-5, atol=1e-6)
