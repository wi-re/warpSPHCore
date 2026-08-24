"""In-process standing test for `warpOperationJVP`'s Covariance branch
(`warpier_unified_operator_wrapper_plan.md` Phase 2): `computeCovarianceGeometryJVP`
(`coreOperations/wp_covarianceJVP.py`) already existed and was already consumed
internally by `renorm.py`'s `computeRenormalizationMatricesJVP`, but was never
registered in `operations.py`'s `_GEOMETRY_JVP_OPERATIONS`/dispatch, so
`warpOperationJVP(Covariance, ...)` wasn't a supported *public* call before this
plan -- this is new public-dispatch coverage, not new math, gated the same way
`test_forward_mode_geometry_jvp_density.py` gates Density's own geometry JVP
branch: a reverse-mode-Jacobian reference on the production `warpOperation
(Covariance)` call.
"""

from __future__ import annotations

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
from warpSPHCore.enumTypes import KernelFunctions, OperationDirection, SupportScheme, WarpOperation

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
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE) * (1.0 + 0.15 * torch.linspace(-1, 1, n, dtype=DTYPE))
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


def _densities(positions, supports, masses, kinds, domain, adjacency):
    p = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    return warpOperation(
        p, OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                               supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    ).detach()


def _check(positions, supports, masses, domain, seed):
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    densities = _densities(positions, supports, masses, kinds, domain, adjacency)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Covariance,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    def f(pos, sup):
        p = ParticleState(positions=pos, supports=sup, masses=masses, densities=densities, kinds=kinds)
        return warpOperation(p, props, domain, adjacency=adjacency)

    pos0 = positions.clone().requires_grad_(True)
    sup0 = supports.clone().requires_grad_(True)

    torch.manual_seed(seed)
    dpos = torch.randn_like(positions)
    dsup = torch.randn_like(supports) * 0.1

    J = torch.autograd.functional.jacobian(f, (pos0, sup0), vectorize=False)
    out = f(pos0, sup0).detach()
    acc = torch.zeros(out.numel(), dtype=DTYPE, device=DEVICE)
    for Jk, vk in zip(J, (dpos, dsup)):
        acc = acc + Jk.reshape(out.numel(), -1) @ vk.reshape(-1)
    reference = acc.reshape(out.shape)

    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    # Self-referencing (referenceParticles defaults to queryParticles), so the
    # same geometry perturbation applies to both roles -- matching f's own
    # single-ParticleState perturbation above.
    assembled = warpOperationJVP(
        p0, props, domain, adjacency=adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
        referenceTangentState=ParticleTangentState(positions=dpos, supports=dsup, masses=None),
    )

    torch.testing.assert_close(assembled, reference, rtol=1e-3, atol=1e-5)


def test_covarianceGeometryJVP_matches_jacobian_reference_1d():
    positions, supports, masses = _line_case()
    _check(positions, supports, masses, _make_domain(dim=1), seed=0)


def test_covarianceGeometryJVP_matches_jacobian_reference_2d():
    positions, supports, masses = _grid_case_2d()
    _check(positions, supports, masses, _make_domain(dim=2), seed=1)


def test_covarianceGeometryJVP_none_adjacency_matches_explicit():
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    densities = _densities(positions, supports, masses, kinds, domain, adjacency)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Covariance,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    torch.manual_seed(2)
    dpos = torch.randn_like(positions)
    tangentState = ParticleTangentState(positions=dpos, supports=None, masses=None)
    viaExplicit = warpOperationJVP(p0, props, domain, adjacency=adjacency,
                                   queryTangentState=tangentState, referenceTangentState=tangentState)
    viaNone = warpOperationJVP(p0, props, domain, adjacency=None,
                               queryTangentState=tangentState, referenceTangentState=tangentState)
    torch.testing.assert_close(viaNone, viaExplicit, rtol=1e-4, atol=1e-5)
