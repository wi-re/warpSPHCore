"""In-process standing test for `warpOperationJVP` (`warpier_forward_mode_plan.md`
Phase 2): asserts it reproduces `scripts/spike_forward_mode_tier1.py`'s value
JVP identity for each of the five value-consuming operators, the same way
`test_gradcheck_scripts.py` gates the spike script itself as a subprocess.

The spike script runs in a subprocess at float64 (precision is baked into
every kernel at first `warpSPHCore` import and cannot change mid-process --
see `test_gradcheck_scripts.py`'s docstring); this file instead runs
in-process alongside the rest of `tests/operations/` at the suite's default
float32, on small hand-built cases so the reverse-mode Jacobian reference
(one backward pass per output element) stays cheap. Looser (float32) but
independent evidence for the same identity, not a replacement for the spike
gate.
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
    # nonzero densities rather than leaving None (spike_forward_mode_tier1.py's
    # compute_densities does the same for the same reason).
    densities = warpOperation(
        particles,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                            supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll),
        domain, adjacency=adjacency,
    )
    particles = ParticleState(positions=positions, supports=supports, masses=masses, densities=densities,
                              kinds=kinds)
    return particles, adjacency


def _make_f(particles, domain, adjacency, operation, gradient_mode=GradientScheme.Naive):
    props = OperationProperties(
        kernel=KERNEL, operation=operation, supportMode=SupportScheme.Gather,
        operationMode=OperationDirection.AllToAll, gradientMode=gradient_mode,
    )

    def f(qval, rval):
        return warpOperation(particles, props, domain, queryValues=qval, referenceValues=rval,
                             adjacency=adjacency)

    return f, props


def _check_jvp_identity(particles, domain, adjacency, operation, value_shape, gradient_mode=GradientScheme.Naive):
    f, props = _make_f(particles, domain, adjacency, operation, gradient_mode)

    torch.manual_seed(0)
    dq = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE)

    # warpOperationJVP delegates to warpOperation on the tangent arrays --
    # exact identity, not merely close.
    jvp_out = warpOperationJVP(particles, props, domain, tangentQueryValues=dq,
                               tangentReferenceValues=dr, adjacency=adjacency)
    direct_out = f(dq, dr)
    torch.testing.assert_close(jvp_out, direct_out, rtol=0, atol=0)

    # Independent reference: the reverse-mode Jacobian (already validated by
    # the gradcheck suite) contracted with the tangent.
    qval = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    rval = torch.randn(*value_shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    J_q, J_r = torch.autograd.functional.jacobian(f, (qval, rval), vectorize=False)
    out_numel = jvp_out.numel()
    in_numel = qval.numel()
    ref = (J_q.reshape(out_numel, in_numel) @ dq.reshape(in_numel)
           + J_r.reshape(out_numel, in_numel) @ dr.reshape(in_numel)).reshape(jvp_out.shape)

    torch.testing.assert_close(jvp_out, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("operation,gradient_mode", [
    (WarpOperation.Interpolate, GradientScheme.Naive),
    (WarpOperation.Gradient, GradientScheme.Naive),
    (WarpOperation.Gradient, GradientScheme.Difference),
    (WarpOperation.Laplacian, GradientScheme.Naive),
])
def test_forwardOperationJVP_matches_jacobian_reference_1d_scalar(operation, gradient_mode):
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    _check_jvp_identity(particles, domain, adjacency, operation, (positions.shape[0],), gradient_mode)


@pytest.mark.parametrize("operation", [WarpOperation.Divergence, WarpOperation.Curl])
def test_forwardOperationJVP_matches_jacobian_reference_2d_vector(operation):
    positions, supports, masses = _grid_case_2d()
    domain = _make_domain(dim=2)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    _check_jvp_identity(particles, domain, adjacency, operation, (positions.shape[0], 2),
                        gradient_mode=GradientScheme.Difference)


def test_forwardOperationJVP_combines_value_and_geometry_tangent():
    # warpier_tier2_combined_jvp_plan.md: a value tangent alongside a
    # geometry tangent is no longer rejected -- it's the sum of the value
    # JVP and geometry JVP paths called independently (each already exact/
    # tested on its own here and in test_forward_mode_geometry_jvp_gradient.py's
    # own jacobian-based "combined" coverage; this is a lighter-weight
    # identity check, not a duplicate of that jacobian derivation).
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Gradient,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                                gradientMode=GradientScheme.Naive)
    n = positions.shape[0]

    torch.manual_seed(0)
    queryValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    referenceValues = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dq = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dr = torch.randn(n, dtype=DTYPE, device=DEVICE)
    dpos = torch.randn_like(positions)

    tier2Only = warpOperationJVP(particles, props, domain, adjacency=adjacency,
                                 tangentQueryPositions=dpos,
                                 queryValues=queryValues, referenceValues=referenceValues)
    tier1Only = warpOperationJVP(particles, props, domain, adjacency=adjacency,
                                 tangentQueryValues=dq, tangentReferenceValues=dr,
                                 queryValues=queryValues, referenceValues=referenceValues)
    combined = warpOperationJVP(particles, props, domain, adjacency=adjacency,
                                tangentQueryPositions=dpos,
                                tangentQueryValues=dq, tangentReferenceValues=dr,
                                queryValues=queryValues, referenceValues=referenceValues)
    torch.testing.assert_close(combined, tier2Only + tier1Only, rtol=0, atol=0)


def test_forwardOperationJVP_rejects_operations_without_a_value_input():
    positions, supports, masses = _line_case()
    domain = _make_domain(dim=1)
    particles, adjacency = _build_adjacency(positions, supports, masses, domain)
    props = OperationProperties(kernel=KERNEL, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)
    n = positions.shape[0]
    with pytest.raises(NotImplementedError, match="value JVP is only defined"):
        warpOperationJVP(particles, props, domain,
                         tangentQueryValues=torch.zeros(n, dtype=DTYPE, device=DEVICE), adjacency=adjacency)
