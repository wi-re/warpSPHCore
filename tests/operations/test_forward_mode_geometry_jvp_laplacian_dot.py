"""In-process standing test for `warpOperationJVP`'s geometry JVP Laplacian
(Dot scheme) branch (`warpier_tier2_jvp_remaining_work_plan.md`'s Dot/Default
follow-up, resolved 2026-08-20): asserts `computeSPHLaplacianDotGeometryJVP`
matches a reverse-mode-Jacobian reference on the production
`warpOperation(Laplacian, laplacianMode=Dot)` call, same pattern as
`test_forward_mode_geometry_jvp_laplacian_brookshaw.py`.

Dot (`computeLaplacianDot2`, `math/wp_laplaciandot.py`, DJ Price SPH/MHD eq
96) projects `dim`-sized blocks of the field against `n_ij`, so unlike
Brookshaw/Naive/Default it needs a genuinely vector-valued field
(`queryValues`/`referenceValues` shape `(n, dim)`) in >1D -- `wp_laplacian.py`
itself rejects a scalar field for this scheme there. 1D is the degenerate
case where a scalar field is still in-scope (block size 1).

**History:** this test originally used finite differences instead of the
usual jacobian reference, because `computeLaplacianDot2`'s own automatic
(`wp.Tape`) reverse-mode adjoint was wrong (a loop-accumulated `proj` value
consumed by a further non-linear op in the same function -- see
`docs/lessons_learned.md`'s "Warp kernel authoring gotchas"). Both
`computeLaplacianDot2` and Tier-2's own `computeSPHLaplacianDotJVP_Func_i`
have since been fixed by moving that reduction into its own `@wp.func`
(`math/wp_laplaciandot.py`'s `computeDotLaplacian(q_ij, n_ij, dim, base)`
overload / `coreOperations/wp_laplacianJVP.py`'s `_laplacianDotProjJVP`) that
*returns* the accumulated value instead of leaving it as a local used in a
non-linear op later in the same function body -- confirmed fixed via
`torch.autograd.gradcheck` (`scripts/gradcheck_tier2_jvp_laplacian.py`) and
via finite differences agreeing with the (now-correct) jacobian reference
used below.
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


def _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme, fieldShape):
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(hash((mode, scheme, fieldShape)) % (2 ** 31))
    queryValues = torch.randn((n, *fieldShape), dtype=DTYPE, device=DEVICE) if fieldShape else torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn((n, *fieldShape), dtype=DTYPE, device=DEVICE) if fieldShape else torch.randn(n, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                supportMode=mode, operationMode=OperationDirection.AllToAll,
                                gradientMode=scheme, laplacianMode=LaplacianScheme.Dot)

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


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric, SupportScheme.KernelMeanSymmetric, SupportScheme.SuperSymmetric])
def test_laplacianDotGeometryJVP_matches_jacobian_reference_1d_scalar(mode, scheme):
    # dim=1: block size 1, so a plain scalar field is still in-scope.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.KernelMeanSymmetric)
    _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme, fieldShape=())


def _check_combined_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme, fieldShape):
    # warpier_tier2_combined_jvp_plan.md: geometry tangent + value tangent
    # together should equal the geometry JVP and value JVP contributions summed.
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    torch.manual_seed(hash(("combined", mode, scheme, fieldShape)) % (2 ** 31))
    queryValues = torch.randn((n, *fieldShape), dtype=DTYPE, device=DEVICE) if fieldShape else torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn((n, *fieldShape), dtype=DTYPE, device=DEVICE) if fieldShape else torch.randn(n, dtype=DTYPE, device=DEVICE)

    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                supportMode=mode, operationMode=OperationDirection.AllToAll,
                                gradientMode=scheme, laplacianMode=LaplacianScheme.Dot)

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
def test_laplacianDotGeometryJVP_combined_matches_jacobian_reference_2d(scheme):
    # dim=2 vector field ((n, 2)): Dot's real intended use, same as the
    # geometry-only 2d vectorField case above.
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    _check_combined_jacobian_reference(positions, supports, masses, domain, adjacency, SupportScheme.Gather, scheme, fieldShape=(2,))


@pytest.mark.parametrize("scheme", list(GradientScheme))
@pytest.mark.parametrize("mode", [SupportScheme.Gather, SupportScheme.MeanSymmetric])
def test_laplacianDotGeometryJVP_matches_jacobian_reference_2d_vectorField(mode, scheme):
    # dim=2: Dot needs a genuinely dim-sized-block field -- a (n, 2) vector
    # field (e.g. velocity), the scheme's real intended use.
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0_forAdjacency = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0_forAdjacency, domain, mode=SupportScheme.Gather)
    _check_jacobian_reference(positions, supports, masses, domain, adjacency, mode, scheme, fieldShape=(2,))


def _minimal_case_2d_vector():
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    n = positions.shape[0]
    kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p0, domain, mode=SupportScheme.Gather)
    densities = _densities_for(positions, supports, masses, kinds, domain, adjacency)
    p0 = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities, kinds=kinds)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Laplacian,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                gradientMode=GradientScheme.Symmetric, laplacianMode=LaplacianScheme.Dot)
    queryValues = torch.randn((n, 2), dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn((n, 2), dtype=DTYPE, device=DEVICE)
    return positions, p0, domain, adjacency, props, queryValues, referenceValues


def test_laplacianDotGeometryJVP_rejects_missing_values():
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case_2d_vector()
    with pytest.raises(ValueError, match="queryValues"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None))


def test_laplacianDotGeometryJVP_rejects_scalarField_in_2d():
    # wp_laplacian.py's own restriction, re-enforced by
    # computeSPHLaplacianDotGeometryJVP: a scalar field has no dim-sized
    # blocks to project in >1D.
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case_2d_vector()
    n = positions.shape[0]
    scalarValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    with pytest.raises(ValueError, match="flattened size"):
        warpOperationJVP(p0, props, domain, adjacency=adjacency,
                         queryTangentState=ParticleTangentState(positions=torch.zeros_like(positions), supports=None, masses=None),
                         queryValues=scalarValues, referenceValues=scalarValues)


def test_laplacianDotGeometryJVP_geometryOnly_unchanged_when_combination_allowed():
    # Regression guard (warpier_tier2_combined_jvp_plan.md step 5): the
    # geometry-only path (no value tangent supplied) must return exactly
    # what it did before the combined path was added.
    positions, p0, domain, adjacency, props, qv, rv = _minimal_case_2d_vector()
    dpos = torch.zeros_like(positions)
    viaJVP = warpOperationJVP(p0, props, domain, adjacency=adjacency,
                              queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None), queryValues=qv, referenceValues=rv)
    from warpSPHCore.coreOperations import computeSPHLaplacianDotGeometryJVP
    direct = computeSPHLaplacianDotGeometryJVP(
        p0, domain, props.kernel, props.supportMode, adjacency,
        queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None), queryValues=qv, referenceValues=rv,
        gradientMode=props.gradientMode,
    )
    torch.testing.assert_close(viaJVP, direct, rtol=0, atol=0)


def test_laplacianDotGeometryJVP_grid_traversal_matches_adjacency_traversal():
    # computeSPHLaplacianDotGeometryJVP (CSR, warpier_tier2_jvp_csr_backend_plan.md)
    # also supports grid (CompactHashMap) traversal -- exercised here via a
    # direct import against the low-level function, independent of
    # warpOperationJVP.
    from warpSPHCore.coreOperations import computeSPHLaplacianDotGeometryJVP

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
    queryValues = torch.randn((n, 2), dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn((n, 2), dtype=DTYPE, device=DEVICE)
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
    viaAdjacency = computeSPHLaplacianDotGeometryJVP(adjacency=adjacency, **common)
    viaGrid = computeSPHLaplacianDotGeometryJVP(adjacency=hashMap, **common)

    torch.testing.assert_close(viaGrid, viaAdjacency, rtol=1e-5, atol=1e-6)
