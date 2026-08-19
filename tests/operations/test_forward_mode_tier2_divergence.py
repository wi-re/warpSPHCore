"""In-process standing test for `warpOperationJVP`'s Tier-2 Divergence
branch (`warpier_tier2_operators_plan.md` Step 5, `warpier_adjoint.md` Tier
2.2): asserts `computeSPHDivergencePositionJVP` matches a reverse-mode-
Jacobian reference on the production `warpOperation(Divergence)` call
(`divergenceDotMode=False`, the only mode in Tier-2 scope), same pattern as
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
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass, tangentQueryDensities=ddensity, tangentReferenceDensities=ddensity,
        queryValues=queryValues, referenceValues=referenceValues,
    )
    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric])
def test_divergencePositionJVP_matches_jacobian_reference_2d(mode, scheme):
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


def test_divergencePositionJVP_rejects_missing_values():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    with pytest.raises(ValueError, match="queryValues"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions))


def test_divergencePositionJVP_rejects_dotMode():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    dotProps = OperationProperties(kernel=KERNEL, operation=WarpOperation.Divergence,
                                   supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                   gradientMode=GradientScheme.Symmetric, divergenceDotMode=True)
    with pytest.raises(NotImplementedError, match="Tier-2"):
        warpOperationJVP(p0, dotProps, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         queryValues=qv, referenceValues=rv)


def test_divergencePositionJVP_rejects_consistentDivergence():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    with pytest.raises(NotImplementedError, match="Tier-2"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         consistentDivergence=True,
                         queryValues=qv, referenceValues=rv)


def test_divergencePositionJVP_rejects_tangentQueryMasses():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case()
    n = positions.shape[0]
    with pytest.raises(NotImplementedError, match="Tier-2"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         tangentQueryMasses=torch.zeros(n, dtype=DTYPE),
                         queryValues=qv, referenceValues=rv)
