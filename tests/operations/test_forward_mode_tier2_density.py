"""In-process standing test for `warpOperationJVP`'s Tier-2 Density branch
(`warpier_forward_mode_plan.md` Phase 4, `warpier_adjoint.md` Tier 2.1):
asserts `computeSPHDensityPositionJVP` (dispatched to for Density's
position/support/mass tangent) matches a reverse-mode-Jacobian reference on
the production `warpOperation(Density)` call, the same reference-construction
pattern `spike_forward_mode_tier1.py`/`spike_forward_mode_tier2_density.py`
use -- see either docstring for why `torch.autograd.functional.jvp` must not
be used as the reference instead (it returns a silently zero tangent through
this bridge).

Runs in-process at the suite's default float32 (unlike the Tier 2.1 spike,
which runs in a float64 subprocess -- precision is baked in at first
`warpSPHCore` import and cannot change mid-process), on small hand-built
cases so the Jacobian reference (one backward pass per output element) stays
cheap. Looser tolerance, independent evidence for the same identity.
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
    # KernelMeanSymmetric to numerically identical cases (warpier_adjoint.md
    # Tier 2.1's "process notes"), silently hiding a wrong SupportScheme
    # dispatch.
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


@pytest.mark.parametrize("mode", [
    SupportScheme.Gather, SupportScheme.Scatter, SupportScheme.MeanSymmetric,
    SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric,
])
def test_densityPositionJVP_matches_jacobian_reference_1d(mode):
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.KernelMeanSymmetric)
    assert isinstance(adjacency, AdjacencyList)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=mode, operationMode=OperationDirection.AllToAll)

    def f(pos, sup, mass):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=None, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency)

    pos0 = positions.clone().requires_grad_(True)
    sup0 = supports.clone().requires_grad_(True)
    mass0 = masses.clone().requires_grad_(True)

    torch.manual_seed(0)
    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1
    dmass = torch.randn_like(masses)

    J = torch.autograd.functional.jacobian(f, (pos0, sup0, mass0), vectorize=False)
    out = f(pos0, sup0, mass0).detach()
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, (dpos, dsup, dmass)):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    reference = acc.reshape(out.shape)

    assembled = warpOperationJVP(
        p0, props, domain, adjacency=adjacency,
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass,
    )

    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


def test_densityPositionJVP_matches_jacobian_reference_2d():
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    def f(pos, sup, mass):
        p = ParticleState(positions=pos, supports=sup, masses=mass, densities=None, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency)

    pos0 = positions.clone().requires_grad_(True)
    sup0 = supports.clone().requires_grad_(True)
    mass0 = masses.clone().requires_grad_(True)

    torch.manual_seed(1)
    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1
    dmass = torch.randn_like(masses)

    J = torch.autograd.functional.jacobian(f, (pos0, sup0, mass0), vectorize=False)
    out = f(pos0, sup0, mass0).detach()
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, (dpos, dsup, dmass)):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    reference = acc.reshape(out.shape)

    assembled = warpOperationJVP(
        p0, props, domain, adjacency=adjacency,
        tangentQueryPositions=dpos, tangentReferencePositions=dpos,
        tangentQuerySupports=dsup, tangentReferenceSupports=dsup,
        tangentReferenceMasses=dmass,
    )

    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


def test_densityPositionJVP_rejects_nonAdjacencyList():
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    n = positions.shape[0]
    with pytest.raises(NotImplementedError, match="AdjacencyList"):
        warpOperationJVP(p0, props, domain, adjacency=None,
                         tangentQueryPositions=torch.zeros_like(positions))


def test_densityPositionJVP_rejects_combination_with_value_tangent():
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    n = positions.shape[0]
    with pytest.raises(NotImplementedError, match="Tier-2"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions),
                         tangentQueryValues=torch.zeros(n, dtype=DTYPE))


def test_otherOperators_tier2_still_raise():
    # Density, Interpolate, Gradient, Divergence, Curl, and Laplacian(Brookshaw)
    # are all implemented now (warpier_tier2_operators_plan.md steps 0-7) --
    # Covariance is the one operator that stays out of Tier-2 scope throughout
    # (no Tier-2 formula was ever derived for it), so it's what's left to prove
    # "still not implemented" here, per the plan's own "Tests" section.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Covariance,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    with pytest.raises(NotImplementedError, match="Tier-2"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         tangentQueryPositions=torch.zeros_like(positions))
