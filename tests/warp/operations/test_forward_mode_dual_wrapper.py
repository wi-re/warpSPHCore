"""`warpier_unified_operator_wrapper_plan.md` Phase 1+2/3 (partial -- both
the Tier-1 value tangent and the Tier-2 geometry tangent, including
Covariance's newly-public dispatch; a dedicated combined-tangent test and
the remaining CRK/renorm-correction combinations are still Phase 3's own
scope). Asserts a caller can wrap a leaf tensor as an ordinary
`torch.autograd.forward_ad` dual tensor and pass it straight into
`warpOperation` -- no separate `warpOperationJVP` call, no
`tangentQueryValues=`/`queryTangentState=`-style parallel kwargs -- and get
back a dual tensor whose tangent matches `warpOperationJVP` bit-for-bit
(same underlying `jvp_fn` delegation; this guards the new
`StateAwareWarpFunction.jvp()` wiring, not the math
`test_forward_mode_value_jvp.py`/`test_forward_mode_geometry_jvp_*.py`
already cover).
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as fwAD

from warpSPHCore import (
    DomainDescription,
    OperationProperties,
    ParticleState,
    ParticleTangentState,
    radiusSearchCompactHashMap,
    warpOperation,
    warpOperationJVP,
)
from warpSPHCore.enumTypes import GradientScheme, KernelFunctions, OperationDirection, SupportScheme, WarpOperation

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
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    return positions, supports, masses


def _grid_case_2d(n_per_side: int = 3, spacing: float = 0.4):
    coords = torch.linspace(-(n_per_side - 1) / 2 * spacing, (n_per_side - 1) / 2 * spacing,
                             n_per_side, dtype=DTYPE, device=DEVICE)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    h = max(2.5 * spacing, 1e-3)
    n = positions.shape[0]
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE)
    return positions, supports, masses


def _build_adjacency(positions, supports, masses, domain):
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    particles = ParticleState(positions=positions, supports=supports, masses=masses, densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)

    # Value operators divide by density internally -- give them realistic,
    # nonzero densities rather than leaving None.
    densities = warpOperation(
        particles,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                            supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    particles = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities,
                              kinds=kinds)
    return particles, adjacency


@pytest.mark.parametrize("operation,gradient_mode", [
    (WarpOperation.Interpolate, GradientScheme.Naive),
    (WarpOperation.Gradient, GradientScheme.Naive),
    (WarpOperation.Gradient, GradientScheme.Difference),
    (WarpOperation.Laplacian, GradientScheme.Naive),
])
def test_dualTensor_valueTangent_matches_warpOperationJVP_1d_scalar(operation, gradient_mode):
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=operation, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=gradient_mode)

    torch.manual_seed(0)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dq = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(n, dtype=DTYPE, device=DEVICE)

    with fwAD.dual_level():
        dual_q = fwAD.make_dual(qval, dq)
        dual_r = fwAD.make_dual(rval, dr)
        out = warpOperation(particles, props, domain, queryValues=dual_q, referenceValues=dual_r,
                            adjacency=adjacency)
        val, tangent = fwAD.unpack_dual(out)

    assert tangent is not None
    torch.testing.assert_close(val, warpOperation(particles, props, domain, queryValues=qval,
                                                    referenceValues=rval, adjacency=adjacency),
                               rtol=0, atol=0)
    ref = warpOperationJVP(particles, props, domain, tangentQueryValues=dq, tangentReferenceValues=dr,
                           queryValues=qval, referenceValues=rval, adjacency=adjacency)
    torch.testing.assert_close(tangent, ref, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [WarpOperation.Divergence, WarpOperation.Curl])
def test_dualTensor_valueTangent_matches_warpOperationJVP_2d_vector(operation):
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=operation, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Difference)

    torch.manual_seed(1)
    qval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    dq = torch.randn(n, 2, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(n, 2, dtype=DTYPE, device=DEVICE)

    with fwAD.dual_level():
        dual_q = fwAD.make_dual(qval, dq)
        dual_r = fwAD.make_dual(rval, dr)
        out = warpOperation(particles, props, domain, queryValues=dual_q, referenceValues=dual_r,
                            adjacency=adjacency)
        _, tangent = fwAD.unpack_dual(out)

    ref = warpOperationJVP(particles, props, domain, tangentQueryValues=dq, tangentReferenceValues=dr,
                           queryValues=qval, referenceValues=rval, adjacency=adjacency)
    torch.testing.assert_close(tangent, ref, rtol=0, atol=0)


def test_dualTensor_onlyOneSideDual_stillMatches():
    # Only referenceValues wrapped as dual -- exercises the "None tangent at
    # a value-eligible position" path in _build_value_jvp_fn (zeros_like
    # fallback), not just "both sides dual".
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)

    torch.manual_seed(2)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(n, dtype=DTYPE, device=DEVICE)

    with fwAD.dual_level():
        dual_r = fwAD.make_dual(rval, dr)
        out = warpOperation(particles, props, domain, queryValues=qval, referenceValues=dual_r,
                            adjacency=adjacency)
        _, tangent = fwAD.unpack_dual(out)

    # Gradient's JVP reads both fi and fj tangents (unlike Interpolate, which
    # never touches queryValues at all) -- an explicit zero stands in for the
    # side that carries no dual tangent in this test.
    ref = warpOperationJVP(particles, props, domain, tangentQueryValues=torch.zeros_like(qval),
                           tangentReferenceValues=dr, queryValues=qval, referenceValues=rval,
                           adjacency=adjacency)
    torch.testing.assert_close(tangent, ref, rtol=0, atol=0)


def test_dualTensor_geometryTangent_raises_notImplemented():
    # Phase 2 wires geometry tangents too -- a query-side *mass* tangent has
    # no JVP formula for any value-having operator though
    # (`warpOperationJVP`'s own "tangentQueryMasses is not None: raise"),
    # exercised here through the dual-tensor path specifically rather than
    # an explicit warpOperationJVP call.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)

    torch.manual_seed(3)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dmasses = torch.randn_like(masses)

    with fwAD.dual_level():
        dual_masses = fwAD.make_dual(masses, dmasses)
        dual_particles = ParticleState(positions=positions, supports=supports, masses=dual_masses,
                                       densities=particles.densities, kinds=particles.kinds)
        with pytest.raises(NotImplementedError, match="tangentQueryMasses"):
            warpOperation(dual_particles, props, domain, queryValues=qval, referenceValues=rval,
                         adjacency=adjacency)


def test_dualTensor_geometryTangent_matches_warpOperationJVP():
    # Phase 2: a position dual tensor drives the geometry-tangent path
    # (previously Phase 1 could only raise here) -- checked against the
    # already-tested explicit warpOperationJVP(queryTangentState=...) call.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Difference)

    torch.manual_seed(5)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)

    with fwAD.dual_level():
        dual_positions = fwAD.make_dual(positions, dpos)
        dual_particles = ParticleState(positions=dual_positions, supports=supports, masses=masses,
                                       densities=particles.densities, kinds=particles.kinds)
        out = warpOperation(dual_particles, props, domain, queryValues=qval, referenceValues=rval,
                            adjacency=adjacency)
        _, tangent = fwAD.unpack_dual(out)

    assert tangent is not None
    # referenceParticles defaults to queryParticles on both the dual call and
    # this reference call, so the dual positions perturb BOTH roles -- the
    # reference-side tangent must match, not stay implicitly zero.
    ref = warpOperationJVP(particles, props, domain, adjacency=adjacency,
                           queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None),
                           referenceTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None),
                           queryValues=qval, referenceValues=rval)
    torch.testing.assert_close(tangent, ref, rtol=0, atol=0)


def test_dualTensor_covarianceGeometryTangent_matches_warpOperationJVP():
    # Covariance (Phase 2: promoted into public warpOperationJVP dispatch,
    # geometry-tangent only -- no value input at all) through the dual-
    # tensor path.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)

    torch.manual_seed(6)
    dpos = torch.randn_like(positions)

    with fwAD.dual_level():
        dual_positions = fwAD.make_dual(positions, dpos)
        dual_particles = ParticleState(positions=dual_positions, supports=supports, masses=masses,
                                       densities=particles.densities, kinds=particles.kinds)
        props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Covariance, supportMode=SupportScheme.Gather,
                                    operationMode=OperationDirection.AllToAll)
        out = warpOperation(dual_particles, props, domain, adjacency=adjacency)
        _, tangent = fwAD.unpack_dual(out)

    assert tangent is not None
    # referenceParticles defaults to queryParticles on both the dual call and
    # this reference call, so the dual positions perturb BOTH roles.
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Covariance, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll)
    ref = warpOperationJVP(particles, props, domain, adjacency=adjacency,
                           queryTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None),
                           referenceTangentState=ParticleTangentState(positions=dpos, supports=None, masses=None))
    torch.testing.assert_close(tangent, ref, rtol=0, atol=0)


def test_dualTensor_chained_backward_reaches_original_leaves():
    # Entirely inside one dual_level: warpOperation -> warpOperation, then
    # .backward() on the unpacked tangent (ordinary reverse-mode, outside
    # the dual_level) reaches the original leaves -- unlike
    # test_gradcheck_tier2_jvp_chained_backprop.py, no explicit
    # warpOperationJVP call anywhere in this test.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Interpolate, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll)

    torch.manual_seed(4)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)
    dr = torch.randn(n, dtype=DTYPE, device=DEVICE, requires_grad=True)

    with fwAD.dual_level():
        dual_r = fwAD.make_dual(rval, dr)
        mid = warpOperation(particles, props, domain, referenceValues=dual_r, adjacency=adjacency)
        out = warpOperation(particles, props, domain, referenceValues=mid, adjacency=adjacency)
        _, tangent = fwAD.unpack_dual(out)

    tangent.sum().backward()
    assert dr.grad is not None and float(dr.grad.abs().max()) > 0
    assert rval.grad is None  # tangent graph must not cross-contaminate the primal leaf


def test_dualTensor_combinedValueAndGeometryTangent_matches_warpOperationJVP():
    # warpier_tier2_combined_jvp_plan.md: a value tangent *and* a geometry
    # tangent live at once is the sum of the two -- exercised here entirely
    # through the dual-tensor path (both queryValues/referenceValues *and*
    # positions wrapped as dual in the same call), matching
    # test_forward_mode_value_jvp.py's explicit-kwarg version of the same
    # identity.
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)

    torch.manual_seed(7)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dq = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)

    with fwAD.dual_level():
        dual_q = fwAD.make_dual(qval, dq)
        dual_r = fwAD.make_dual(rval, dr)
        dual_positions = fwAD.make_dual(positions, dpos)
        dual_particles = ParticleState(positions=dual_positions, supports=supports, masses=masses,
                                       densities=particles.densities, kinds=particles.kinds)
        out = warpOperation(dual_particles, props, domain, queryValues=dual_q, referenceValues=dual_r,
                            adjacency=adjacency)
        _, tangent = fwAD.unpack_dual(out)

    assert tangent is not None
    tangentState = ParticleTangentState(positions=dpos, supports=None, masses=None)
    ref = warpOperationJVP(particles, props, domain, adjacency=adjacency,
                           queryTangentState=tangentState, referenceTangentState=tangentState,
                           tangentQueryValues=dq, tangentReferenceValues=dr,
                           queryValues=qval, referenceValues=rval)
    torch.testing.assert_close(tangent, ref, rtol=0, atol=0)


@pytest.mark.parametrize("operation,gradient_mode", [
    (WarpOperation.Gradient, GradientScheme.Difference),
    (WarpOperation.Divergence, GradientScheme.Difference),
    (WarpOperation.Curl, GradientScheme.Difference),
    (WarpOperation.Laplacian, GradientScheme.Naive),
])
def test_dualTensor_geometryTangent_matches_warpOperationJVP_perOperator(operation, gradient_mode):
    # Phase 3: geometry tangent on every operator _build_geometry_jvp_fn is
    # wired for, not just Gradient -- each of Divergence/Curl/Laplacian has
    # its own outputShape formula (the exact thing _reflattenResult had to
    # correct for at the *value*-tangent level for Laplacian/Divergence
    # specifically), so this is the one place that shape-correctness claim
    # gets checked for the *geometry* tangent too, for every operator.
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=operation, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=gradient_mode)

    torch.manual_seed(8)
    valueShape = (n, 2) if operation in (WarpOperation.Divergence, WarpOperation.Curl) else (n,)
    qval = torch.randn(*valueShape, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(*valueShape, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)

    with fwAD.dual_level():
        dual_positions = fwAD.make_dual(positions, dpos)
        dual_particles = ParticleState(positions=dual_positions, supports=supports, masses=masses,
                                       densities=particles.densities, kinds=particles.kinds)
        out = warpOperation(dual_particles, props, domain, queryValues=qval, referenceValues=rval,
                            adjacency=adjacency)
        _, tangent = fwAD.unpack_dual(out)

    assert tangent is not None
    tangentState = ParticleTangentState(positions=dpos, supports=None, masses=None)
    ref = warpOperationJVP(particles, props, domain, adjacency=adjacency,
                           queryTangentState=tangentState, referenceTangentState=tangentState,
                           queryValues=qval, referenceValues=rval)
    torch.testing.assert_close(tangent, ref, rtol=0, atol=0)


def test_dualTensor_crkTangent_wiring_does_not_raise():
    # CRK tangent passthrough: sphCtx.corrections.crk is captured by closure
    # (a Python object, not reconstructed from flat tensors) and
    # crkTangentState is built straight from flat positions 15-18 -- no
    # shape ambiguity like the value-tangent case, so this checks the wiring
    # reaches warpOperationJVP's already-tested CRK path at all, the same
    # "dummy zero state, assert finite" pattern
    # test_forward_mode_geometry_jvp_gradient.py's own
    # test_gradientGeometryJVP_accepts_crkState uses.
    from warpSPHCore.dataTypes import CRKState

    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)

    torch.manual_seed(9)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)
    crkA = torch.zeros(n, dtype=DTYPE, device=DEVICE)
    dcrkA = torch.randn(n, dtype=DTYPE, device=DEVICE)
    crkB = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
    crkGradA = torch.zeros(n, 1, dtype=DTYPE, device=DEVICE)
    crkGradB = torch.zeros(n, 1, 1, dtype=DTYPE, device=DEVICE)

    with fwAD.dual_level():
        dual_positions = fwAD.make_dual(positions, dpos)
        dual_crkA = fwAD.make_dual(crkA, dcrkA)
        dual_particles = ParticleState(positions=dual_positions, supports=supports, masses=masses,
                                       densities=particles.densities, kinds=particles.kinds)
        crkState = CRKState(A=dual_crkA, B=crkB, gradA=crkGradA, gradB=crkGradB)
        out = warpOperation(dual_particles, props, domain, queryValues=qval, referenceValues=rval,
                            adjacency=adjacency, crkState=crkState)
        _, tangent = fwAD.unpack_dual(out)

    assert tangent is not None and torch.isfinite(tangent).all()


def test_dualTensor_renormTangent_wiring_does_not_raise():
    # Same wiring-only check as the CRK test above, for the gradient-
    # renormalization tangent (flat position 10).
    from warpSPHCore.dataTypes import RenormalizationState

    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    n = positions.shape[0]
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient, supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.AllToAll, gradientMode=GradientScheme.Naive)

    torch.manual_seed(10)
    qval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    rval = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)
    renormMat = torch.eye(1, dtype=DTYPE, device=DEVICE).expand(n, 1, 1).contiguous()
    drenormMat = torch.randn(n, 1, 1, dtype=DTYPE, device=DEVICE)

    with fwAD.dual_level():
        dual_positions = fwAD.make_dual(positions, dpos)
        dual_renormMat = fwAD.make_dual(renormMat, drenormMat)
        dual_particles = ParticleState(positions=dual_positions, supports=supports, masses=masses,
                                       densities=particles.densities, kinds=particles.kinds)
        renormState = RenormalizationState(renormalizationMatrices=dual_renormMat)
        out = warpOperation(dual_particles, props, domain, queryValues=qval, referenceValues=rval,
                            adjacency=adjacency, renormalizationState=renormState)
        _, tangent = fwAD.unpack_dual(out)

    assert tangent is not None and torch.isfinite(tangent).all()
