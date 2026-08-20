"""In-process standing test for `warpOperationJVP`'s geometry JVP Interpolate
branch (`warpier_tier2_operators_plan.md` Step 2, `warpier_adjoint.md` Tier
2.1): asserts `computeSPHInterpolateGeometryJVP` matches a reverse-mode-
Jacobian reference on the production `warpOperation(Interpolate)` call, the
same reference-construction pattern
`test_forward_mode_geometry_jvp_density.py`/`spike_forward_mode_tier2_*.py` use --
see either for why `torch.autograd.functional.jvp` must not be used as the
reference instead (it returns a silently zero tangent through this bridge).

`referenceValues` (`fj`) is frozen throughout the geometry-JVP-only cases below --
only positions/supports/masses/densities are differentiated. The "combined"
case exercises the sum of this geometry JVP contribution with the value JVP
`tangentReferenceValues` contribution
(`warpier_tier2_combined_jvp_plan.md`), differentiating w.r.t. every input
at once.
"""

from __future__ import annotations

import pytest
import torch

from warpSPHCore import (
    AdjacencyList,
    DomainDescription,
    OperationProperties,
    ParticleState,
    radiusSearchCompactHashMap,
    warpOperation,
    warpOperationJVP,
)
from warpSPHCore.enumTypes import OperationDirection, SupportScheme, WarpOperation, KernelFunctions

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
    # Non-uniform supports: uniform h collapses Gather/Scatter/MeanSymmetric/
    # KernelMeanSymmetric to numerically identical cases, silently hiding a
    # wrong SupportScheme dispatch (same rationale as the Density test).
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


@pytest.mark.parametrize("mode", [
    SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric,
    SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric,
])
def test_interpolateGeometryJVP_matches_jacobian_reference_1d(mode):
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.KernelMeanSymmetric)
    assert isinstance(adjacency, AdjacencyList)

    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(0)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Interpolate,
                                supportMode=mode, operationMode=OperationDirection.AllToAll)

    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency, referenceValues=referenceValues)

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
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass, tangentReferenceDensities=ddensity,
        referenceValues=referenceValues,
    )

    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


def test_interpolateGeometryJVP_matches_jacobian_reference_2d():
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)

    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(1)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Interpolate,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    def f(pos, sup, mass, density):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency, referenceValues=referenceValues)

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
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass, tangentReferenceDensities=ddensity,
        referenceValues=referenceValues,
    )

    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


def _minimal_case():
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Interpolate,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    return positions, p0, domain, adjacency, props, referenceValues


def test_interpolateGeometryJVP_none_adjacency_matches_explicit():
    # warpOperationJVP builds a CompactHashMap when adjacency=None, same as
    # warpOperation's own primal path (autograd/arg_extract.py Section 4) --
    # the CSR-ported geometry JVP kernels traverse a grid directly, so this
    # is no longer restricted to a pre-built AdjacencyList.
    positions, p0, domain, adjacency, props, referenceValues = _minimal_case()
    torch.manual_seed(4)
    dpos = torch.randn_like(positions)
    viaExplicit = warpOperationJVP(p0, props, domain, adjacency=adjacency,
                                   tangentQueryPositions=dpos, referenceValues=referenceValues)
    viaNone = warpOperationJVP(p0, props, domain, adjacency=None,
                               tangentQueryPositions=dpos, referenceValues=referenceValues)
    torch.testing.assert_close(viaNone, viaExplicit, rtol=1e-4, atol=1e-5)


def test_interpolateGeometryJVP_combined_matches_jacobian_reference():
    # warpier_tier2_combined_jvp_plan.md: a geometry tangent alongside a
    # value tangent is the sum of the geometry JVP and value JVP contributions, not
    # rejected -- verified against a jacobian differentiating w.r.t. every
    # input (positions/supports/masses/densities/referenceValues) at once.
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)

    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(3)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Interpolate,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    def f(pos, sup, mass, density, rval):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=density, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency, referenceValues=rval)

    pos0 = positions.clone().requires_grad_(True)
    sup0 = supports.clone().requires_grad_(True)
    mass0 = masses.clone().requires_grad_(True)
    density0 = densities.clone().requires_grad_(True)
    rval0 = referenceValues.clone().requires_grad_(True)

    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1
    dmass = torch.randn_like(masses)
    ddensity = torch.randn_like(densities) * 0.1
    drval = torch.randn_like(referenceValues)

    J = torch.autograd.functional.jacobian(f, (pos0, sup0, mass0, density0, rval0), vectorize=False)
    out = f(pos0, sup0, mass0, density0, rval0).detach()
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, (dpos, dsup, dmass, ddensity, drval)):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    reference = acc.reshape(out.shape)

    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    assembled = warpOperationJVP(
        p0, props, domain, adjacency=adjacency,
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass, tangentReferenceDensities=ddensity,
        tangentReferenceValues=drval,
        referenceValues=referenceValues,
    )

    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


def test_interpolateGeometryJVP_geometryOnly_unchanged_when_combination_allowed():
    # Regression guard (warpier_tier2_combined_jvp_plan.md step 5): the
    # geometry-only path (no value tangent supplied) must return exactly
    # what it did before the combined path was added.
    positions, p0, domain, adjacency, props, referenceValues = _minimal_case()
    dpos = torch.zeros_like(positions)
    viaJVP = warpOperationJVP(p0, props, domain, adjacency=adjacency,
                              tangentQueryPositions=dpos, referenceValues=referenceValues)
    from warpSPHCore.coreOperations import computeSPHInterpolateGeometryJVP
    direct = computeSPHInterpolateGeometryJVP(
        p0, domain, props.kernel, props.supportMode, adjacency,
        tangentQueryPositions=dpos, referenceValues=referenceValues,
    )
    torch.testing.assert_close(viaJVP, direct, rtol=0, atol=0)


def test_interpolateGeometryJVP_rejects_missing_referenceValues():
    positions, p0, domain, adjacency, props, referenceValues = _minimal_case()
    with pytest.raises(ValueError, match="referenceValues"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions))


def test_interpolateGeometryJVP_rejects_queryValues():
    positions, p0, domain, adjacency, props, referenceValues = _minimal_case()
    n = positions.shape[0]
    with pytest.raises(ValueError, match="queryValues"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         queryValues=torch.zeros(n, dtype=DTYPE),
                         referenceValues=referenceValues)


def test_interpolateGeometryJVP_rejects_tangentQueryMasses():
    positions, p0, domain, adjacency, props, referenceValues = _minimal_case()
    n = positions.shape[0]
    with pytest.raises(NotImplementedError, match="geometry JVP"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         tangentQueryMasses=torch.zeros(n, dtype=DTYPE),
                         referenceValues=referenceValues)


def test_interpolateGeometryJVP_rejects_volumes():
    positions, p0, domain, adjacency, props, referenceValues = _minimal_case()
    n = positions.shape[0]
    with pytest.raises(NotImplementedError, match="geometry JVP"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         queryVolumes=torch.zeros(n, dtype=DTYPE),
                         referenceValues=referenceValues)


def test_interpolateGeometryJVP_grid_traversal_matches_adjacency_traversal():
    # computeSPHInterpolateGeometryJVP (CSR, warpier_tier2_jvp_csr_backend_plan.md)
    # also supports grid (CompactHashMap) traversal -- exercised here via a
    # direct import against the low-level function (warpOperationJVP itself
    # also accepts a CompactHashMap now, see
    # test_interpolateGeometryJVP_none_adjacency_matches_explicit).
    from warpSPHCore.coreOperations import computeSPHInterpolateGeometryJVP

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
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1
    dmass = torch.randn_like(masses)
    ddensity = torch.randn_like(densities) * 0.1

    common = dict(
        queryParticles=p0, domain=domain, kernel=KERNEL, supportMode=SupportScheme.Gather,
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass, tangentReferenceDensities=ddensity,
        referenceValues=referenceValues,
    )
    viaAdjacency = computeSPHInterpolateGeometryJVP(adjacency=adjacency, **common)
    viaGrid = computeSPHInterpolateGeometryJVP(adjacency=hashMap, **common)

    torch.testing.assert_close(viaGrid, viaAdjacency, rtol=1e-5, atol=1e-6)
