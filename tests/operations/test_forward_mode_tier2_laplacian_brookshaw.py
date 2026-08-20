"""In-process standing test for `warpOperationJVP`'s Tier-2 Laplacian
(Brookshaw scheme) branch (`warpier_tier2_operators_plan.md` Step 7,
`warpier_adjoint.md` Tier 2.2): asserts `computeSPHLaplacianBrookshawPositionJVP`
matches a reverse-mode-Jacobian reference on the production
`warpOperation(Laplacian, laplacianMode=Brookshaw)` call, same pattern as
`test_forward_mode_tier2_gradient.py`.
"""

from __future__ import annotations

import pytest
import torch

from warpSPHCore import (
    DomainDescription,
    OperationProperties,
    ParticleState,
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
                                gradientMode=scheme, laplacianMode=LaplacianScheme.Brookshaw)

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
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass, tangentQueryDensities=ddensity, tangentReferenceDensities=ddensity,
        queryValues=queryValues, referenceValues=referenceValues,
    )
    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric])
def test_laplacianBrookshawPositionJVP_matches_jacobian_reference_1d(mode, scheme):
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.KernelMeanSymmetric)
    _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme)


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric])
def test_laplacianBrookshawPositionJVP_matches_jacobian_reference_2d(mode, scheme):
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme)


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
                                gradientMode=GradientScheme.Symmetric, laplacianMode=LaplacianScheme.Brookshaw)
    queryValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    return positions, p0, domain, adjacency, props, queryValues, referenceValues


def test_laplacianBrookshawPositionJVP_rejects_missing_values():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    with pytest.raises(ValueError, match="queryValues"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions))


@pytest.mark.parametrize("mode", [LaplacianScheme.Dot, LaplacianScheme.Default])
def test_laplacianPositionJVP_rejects_unimplemented_modes(mode):
    # Naive is implemented too now (test_forward_mode_tier2_laplacian_naive.py) --
    # only Dot/Default remain genuinely out of Tier-2 scope.
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    otherProps = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                     supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                     gradientMode=GradientScheme.Symmetric, laplacianMode=mode)
    with pytest.raises(NotImplementedError, match="Tier-2"):
        warpOperationJVP(p0, otherProps, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         queryValues=qv, referenceValues=rv)


def test_laplacianBrookshawPositionJVP_rejects_positiveDivergence():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    posDivProps = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                      supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                      gradientMode=GradientScheme.Symmetric, laplacianMode=LaplacianScheme.Brookshaw,
                                      positiveDivergence=True)
    with pytest.raises(NotImplementedError, match="Tier-2"):
        warpOperationJVP(p0, posDivProps, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         queryValues=qv, referenceValues=rv)


def test_laplacianBrookshawPositionJVP_grid_traversal_matches_adjacency_traversal():
    # computeSPHLaplacianBrookshawPositionJVP (CSR, warpier_tier2_jvp_csr_backend_plan.md)
    # also supports grid (CompactHashMap) traversal, unlike warpOperationJVP's
    # own centralized AdjacencyList-only gate -- exercised here via a direct
    # import.
    from warpSPHCore.coreOperations import computeSPHLaplacianBrookshawPositionJVP

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
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass,
        tangentQueryDensities=ddensity, tangentReferenceDensities=ddensity,
        queryValues=queryValues, referenceValues=referenceValues,
        gradientMode=GradientScheme.Symmetric,
    )
    viaAdjacency = computeSPHLaplacianBrookshawPositionJVP(adjacency=adjacency, **common)
    viaGrid = computeSPHLaplacianBrookshawPositionJVP(adjacency=hashMap, **common)

    torch.testing.assert_close(viaGrid, viaAdjacency, rtol=1e-5, atol=1e-6)
