"""Declared modes, independent singular solves and original-source certificates."""

import pytest
import torch
from torch.autograd import forward_ad

from warpSPHCore.linalg import (
    CGWorkspace, DiagonalPreconditioner, LinearOperator, NullspaceSystem,
    RestrictedSystem, cg, gmres,
)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def operator(A, adjoint=True):
    return LinearOperator(A.shape, A.dtype, A.device, lambda v: A @ v,
                          adjoint=(lambda v: A.T @ v) if adjoint else None)


def graph(device, dtype, sizes=(3, 4)):
    matrices, columns = [], []
    n, start = sum(sizes), 0
    for size in sizes:
        # Independent incidence construction; no projector appears in A.
        incidence = torch.zeros(size - 1, size, dtype=torch.float64)
        for i in range(size - 1):
            incidence[i, i], incidence[i, i + 1] = -1, 1
        weights = torch.linspace(1., 2., size - 1, dtype=torch.float64)
        matrices.append(incidence.T @ torch.diag(weights) @ incidence)
        q = torch.zeros(n, dtype=torch.float64)
        q[start:start + size] = size**-.5
        columns.append(q)
        start += size
    A = torch.block_diag(*matrices).to(device=device, dtype=dtype)
    Q = torch.stack(columns, dim=1).to(device=device, dtype=dtype)
    return A, Q


def close(actual, reference):
    tol = 4e-5 if actual.dtype == torch.float32 else 5e-12
    torch.testing.assert_close(actual, reference.to(actual.dtype), rtol=tol, atol=tol)


@pytest.mark.parametrize('sizes', [(5,), (3, 4)])
@pytest.mark.parametrize('native', [False, True])
def test_compatible_graph_solve_and_all_gauges(device, dtype, sizes, native):
    A, Q = graph(device, dtype, sizes)
    system = NullspaceSystem(operator(A), Q, Q)
    exact = torch.arange(A.shape[0], device=device, dtype=dtype).sin()
    b = A @ exact
    prepared = system.prepareRhs(b)
    assert prepared.compatible and torch.equal(prepared.rhs, b)
    assert prepared.removed.count_nonzero() == 0
    inverse = DiagonalPreconditioner(A.diag(), requireSPD=True)
    precond = system.projectPreconditioner(inverse.asLinearOperator())
    # A nonconstant diagonal does not preserve the mean-free subspace alone.
    test = system.projectLeft(exact)
    assert (Q.T @ inverse(test)).norm() > .01
    close(Q.T @ precond(test), torch.zeros(system.modes, device=device, dtype=dtype))
    warm = system.projectRight(exact * .2 + Q.sum(dim=1) * 7)
    tol = 3e-5 if dtype == torch.float32 else 2e-11
    workspace = CGWorkspace(b, 64, history=False) if native else None
    result = cg(system, prepared.rhs, warm, precond=precond, maxiter=64, rtol=tol,
                workspace=workspace, matvecOut=system.apply if native else None)
    assert result.statusInt() >= 0
    x = system.gauge(result.x)
    reference = torch.linalg.pinv(A.double(), hermitian=True) @ b.double()
    close(x, reference)
    report = system.verify(x, prepared, rtol=tol * 2)
    assert report.requestedConverged and report.solvedConverged and report.projectedConverged
    torch.testing.assert_close(report.requestedResidual, (b - A @ x).norm(), rtol=0, atol=0)
    close(Q.T @ x, torch.zeros(system.modes, device=device, dtype=dtype))
    weights = Q.abs() * torch.arange(1., A.shape[0] + 1, device=device, dtype=dtype)[:, None]
    values = torch.arange(1., system.modes + 1, device=device, dtype=dtype)
    weighted = system.weightedGauge(x, weights, values)
    close((weights / weights.sum(dim=0)).T @ weighted, values)
    indices = torch.tensor([0] if len(sizes) == 1 else [0, sizes[0]], device=device)
    pinned = system.pinGauge(x, indices, values)
    close(pinned[indices], values)
    for adjusted in (weighted, pinned, system.gauge(x, values)):
        close(A @ adjusted, b)
        assert system.verify(adjusted, prepared, rtol=tol * 3).requestedConverged


def test_explicit_projection_never_certifies_the_requested_source(device, dtype):
    A, Q = graph(device, dtype)
    system = NullspaceSystem(operator(A), Q, Q)
    b = A @ torch.arange(A.shape[0], device=device, dtype=dtype).cos() + Q[:, 0] * 2
    with pytest.raises(ValueError, match='incompatible'):
        system.prepareRhs(b)
    prepared = system.prepareRhs(b, policy='project')
    assert not prepared.compatible
    close(prepared.removed, Q[:, 0] * 2)
    close(prepared.incompatibleNorm, b.new_tensor(2.))
    tol = 4e-5 if dtype == torch.float32 else 1e-11
    result = cg(system, prepared.rhs, rtol=tol, maxiter=64)
    assert result.statusInt() >= 0
    report = system.verify(system.gauge(result.x), prepared, rtol=tol * 2)
    assert report.solvedConverged and report.projectedConverged
    assert not report.requestedConverged and report.requestedResidual > 1.9
    close(report.removedNorm, b.new_tensor(2.))


def test_tolerated_component_is_not_silently_removed(device, dtype):
    A = torch.diag(torch.tensor([2., 0.], device=device, dtype=dtype))
    Q = torch.tensor([[0.], [1.]], device=device, dtype=dtype)
    system = NullspaceSystem(operator(A), Q, Q)
    b = torch.tensor([1., 1e-4], device=device, dtype=dtype)
    with pytest.raises(ValueError, match='incompatible'):
        system.prepareRhs(b, rtol=0, atol=0)
    prepared = system.prepareRhs(b, atol=1e-3)
    assert prepared.compatible and torch.equal(prepared.rhs, b)
    report = system.verify(torch.tensor([.5, 0.], device=device, dtype=dtype), prepared, rtol=1e-6)
    assert not report.requestedConverged and not report.projectedConverged


def test_near_zero_projected_source_and_nullspace_warm_start(device, dtype):
    A, Q = graph(device, dtype)
    system = NullspaceSystem(operator(A), Q, Q)
    b = torch.ones(A.shape[0], device=device, dtype=dtype)
    prepared = system.prepareRhs(b, policy='project')
    # A purely incompatible source can leave a rounding-sized projected RHS.
    # Use an explicit absolute residual tolerance rather than silently zeroing it.
    atol = 64 * torch.finfo(dtype).eps * b.norm().item()
    assert prepared.rhs.norm() <= atol and not prepared.compatible
    warm = Q @ Q.new_tensor([4., -2.])
    result = cg(system, prepared.rhs, warm, rtol=0, atol=atol, maxiter=8)
    assert result.statusInt() >= 0 and result.iterations == 0
    torch.testing.assert_close(result.x, warm, rtol=0, atol=0)
    report = system.verify(system.gauge(result.x), prepared, rtol=0, atol=atol)
    assert report.solvedConverged and report.projectedConverged and not report.requestedConverged


def test_nonsymmetric_distinct_left_and_right_spaces(device, dtype):
    # A = u v.T, right kernel = v-perp, left kernel = u-perp.
    A = torch.tensor([[2., 0.], [1., 0.]], device=device, dtype=dtype)
    right = torch.tensor([[0.], [1.]], device=device, dtype=dtype)
    left = torch.tensor([[-1.], [2.]], device=device, dtype=dtype) / 5**.5
    system = NullspaceSystem(operator(A), right, left)
    v = torch.tensor([2., 3.], device=device, dtype=dtype)
    w = v.cos()
    PL = torch.eye(2, device=device, dtype=dtype) - left @ left.T
    PR = torch.eye(2, device=device, dtype=dtype) - right @ right.T
    projected = PL @ A @ PR
    close(system(v), projected @ v)
    close(system.applyAdjoint(w), projected.T @ w)
    close(torch.dot(system(v), w), torch.dot(v, system.applyAdjoint(w)))
    b = torch.tensor([4., 2.], device=device, dtype=dtype)
    prepared = system.prepareRhs(b)
    tol = 3e-5 if dtype == torch.float32 else 1e-11
    # Map left residuals to right solution coordinates via explicit inverse action.
    P = operator(torch.eye(2, device=device, dtype=dtype))
    inverse = system.projectPreconditioner(P)
    close(inverse(v), PR @ PL @ v)
    close(inverse.applyAdjoint(v), PL @ PR @ v)
    result = gmres(system, prepared.rhs, precond=inverse, rtol=tol, maxiter=16)
    assert result.statusInt() >= 0
    x = system.gauge(result.x)
    close(x, torch.linalg.pinv(A.double()) @ b.double())
    assert system.verify(x, prepared, rtol=tol * 2).requestedConverged
    with pytest.raises(ValueError, match='left basis'):
        NullspaceSystem(operator(A), right, right)


def test_inexact_modes_are_exposed_by_original_verification(device, dtype):
    A = torch.diag(torch.tensor([1., 1e-4], device=device, dtype=dtype))
    Q = torch.tensor([[0.], [1.]], device=device, dtype=dtype)
    with pytest.raises(ValueError, match='nullTolerance'):
        NullspaceSystem(operator(A), Q, Q, nullTolerance=1e-6)
    system = NullspaceSystem(operator(A), Q, Q, nullTolerance=1e-3)
    prepared = system.prepareRhs(torch.tensor([1., 0.], device=device, dtype=dtype))
    x = system.gauge(torch.tensor([1., 0.], device=device, dtype=dtype), Q.new_tensor([1e5]))
    report = system.verify(x, prepared, rtol=1e-5)
    assert report.projectedConverged and not report.requestedConverged and not report.solvedConverged
    close(report.requestedResidual, A.new_tensor(10.))


def test_restriction_requires_a_new_nullspace_declaration(device, dtype):
    A, Q = graph(device, dtype, (4,))
    free = torch.tensor([1, 2, 3], device=device)
    reduced = RestrictedSystem(operator(A), free)
    stale = Q[free] / Q[free].norm()
    with pytest.raises(ValueError, match='right basis'):
        NullspaceSystem(reduced.asLinearOperator(), stale, stale)
    empty = torch.empty((3, 0), device=device, dtype=dtype)
    system = NullspaceSystem(reduced.asLinearOperator(), empty, empty)
    b = torch.ones(3, device=device, dtype=dtype)
    result = cg(system, system.prepareRhs(b).rhs, maxiter=16)
    assert result.statusInt() >= 0 and system.verify(result.x, system.prepareRhs(b)).requestedConverged


@pytest.mark.parametrize('n,k', [(0, 0), (3, 0), (3, 3)])
def test_empty_absent_and_full_nullspaces(device, dtype, n, k):
    A = torch.zeros((n, n), device=device, dtype=dtype) if k else torch.eye(n, device=device, dtype=dtype)
    Q = torch.eye(n, device=device, dtype=dtype)[:, :k]
    system = NullspaceSystem(operator(A), Q, Q)
    x = torch.arange(n, device=device, dtype=dtype)
    b = A @ x
    prepared = system.prepareRhs(b)
    result = cg(system, prepared.rhs, x, maxiter=8)
    assert result.statusInt() >= 0
    close(system.gauge(result.x), torch.zeros_like(x) if k else x)
    assert system.verify(result.x, prepared).requestedConverged
    close(system.weightedGauge(x, Q), system.gauge(x))


def test_ownership_revalidation_and_basis_completeness_limit(device, dtype):
    A, Q = graph(device, dtype)
    system = NullspaceSystem(operator(A), Q, Q)
    basis = Q.clone()
    Q.zero_()
    system.leftBasis.zero_()
    system.rightBasis.zero_()
    close(system.rightBasis, basis)
    b = A @ torch.arange(A.shape[0], device=device, dtype=dtype)
    prepared = system.prepareRhs(b)
    saved = prepared.requested.clone()
    b.zero_()
    close(prepared.requested, saved)
    prepared.rhs.zero_()
    close(prepared.requested, saved)
    assert system.validate() is system and system.asLinearOperator() is system.asLinearOperator()
    A.add_(torch.eye(A.shape[0], device=device, dtype=dtype))
    with pytest.raises(ValueError, match='right basis'):
        system.validate()
    # An incomplete declaration cannot be detected by matrix-free mode tests.
    A, Q = graph(device, dtype)
    incomplete = NullspaceSystem(operator(A), Q[:, :1], Q[:, :1])
    missed = incomplete.prepareRhs(Q[:, 1])
    assert missed.compatible
    result = gmres(incomplete, missed.rhs, maxiter=16)
    assert not incomplete.verify(result.x, missed, rtol=1e-4).requestedConverged


def test_invalid_bases_and_tolerances():
    A = torch.diag(torch.tensor([1., 0., 0.]))
    Q = torch.eye(3)[:, 1:]
    for bad in (Q[:, 0], Q.double(), Q.long(), torch.zeros_like(Q), Q * 2,
                Q.clone().fill_(float('nan')), Q.clone().requires_grad_(), Q[:, [0, 0]]):
        with pytest.raises(ValueError):
            NullspaceSystem(operator(A), bad, Q)
    with forward_ad.dual_level():
        with pytest.raises(ValueError, match='differentiate'):
            NullspaceSystem(operator(A), forward_ad.make_dual(Q, Q), Q)
    with pytest.raises(ValueError, match='same number'):
        NullspaceSystem(operator(A), Q, Q[:, :1])
    with pytest.raises(ValueError, match='adjoint'):
        NullspaceSystem(operator(A, False), Q, Q)
    for kw in ({'basisTolerance': -1}, {'basisTolerance': 1}, {'nullTolerance': float('nan')}, {'nullTolerance': True}):
        with pytest.raises(ValueError):
            NullspaceSystem(operator(A), Q, Q, **kw)
    with pytest.raises(ValueError, match='square'):
        NullspaceSystem(operator(A[:2]), Q, Q)


def test_gauge_validation_and_underdetermined_pins(device, dtype):
    A, Q = graph(device, dtype)
    system = NullspaceSystem(operator(A), Q, Q)
    v = torch.ones(A.shape[0], device=device, dtype=dtype)
    # Both pins in one connected component cannot fix the other constant mode.
    with pytest.raises(ValueError, match='independently'):
        system.pinGauge(v, torch.tensor([0, 1], device=device))
    for pins in (torch.tensor([0, 0], device=device), torch.tensor([-1, 3], device=device),
                 torch.tensor([0], device=device), torch.tensor([0., 3.], device=device)):
        with pytest.raises(ValueError):
            system.pinGauge(v, pins)
    for weights in (torch.zeros_like(Q), -Q, Q * float('nan'), Q[:, :1], torch.ones_like(Q)):
        with pytest.raises(ValueError):
            system.weightedGauge(v, weights)
    for values in (v, v[:2].long(), v[:2] * float('inf')):
        with pytest.raises(ValueError):
            system.gauge(v, values)
    with pytest.raises(ValueError, match='finite'):
        system.gauge(v * float('nan'))
    weights = Q.abs() * torch.finfo(dtype).max / 2
    close(system.weightedGauge(v, weights), torch.zeros_like(v))


def test_vector_output_policy_and_report_validation():
    A = torch.diag(torch.tensor([2., 0.]))
    Q = torch.tensor([[0.], [1.]])
    system = NullspaceSystem(operator(A), Q, Q)
    b = torch.tensor([1., 0.])
    prepared = system.prepareRhs(b)
    for call in (system, system.projectRight, system.projectLeft, system.applyAdjoint):
        for bad in (b.double(), b[:, None], b.long()):
            with pytest.raises(ValueError):
                call(bad)
    for call in (system.apply, system.projectRight, system.projectLeft):
        with pytest.raises(ValueError, match='overlap'):
            call(b, b)
        destination = torch.full((4,), float('nan'))[::2]
        assert call(b, destination) is destination
        close(destination, call(b))
    for kwargs in ({'policy': 'auto'}, {'rtol': -1}, {'atol': float('nan')}, {'rtol': True}):
        with pytest.raises(ValueError):
            system.prepareRhs(b, **kwargs)
    for bad in (b * float('nan'), b.new_tensor([1e30, 0.])):
        with pytest.raises(ValueError, match='finite'):
            system.prepareRhs(bad)
    with pytest.raises(ValueError, match='this NullspaceSystem'):
        system.verify(b, NullspaceSystem(operator(A), Q, Q).prepareRhs(b))
    with pytest.raises(ValueError):
        system.projectPreconditioner(operator(A.double()))
    noAdjoint = system.projectPreconditioner(operator(A, False))
    assert noAdjoint.adjoint is None
    with pytest.raises(ValueError, match='no adjoint'):
        noAdjoint.applyAdjoint(b)
    assert not system.verify(b * float('nan'), prepared).requestedConverged


def test_projection_and_action_vector_ad(device, dtype):
    A, Q = graph(device, dtype)
    system = NullspaceSystem(operator(A), Q, Q)
    v = torch.arange(A.shape[0], device=device, dtype=dtype).requires_grad_()
    w = v.detach().sin()
    for action, adjoint in ((system, system.applyAdjoint), (system.projectRight, system.projectRight)):
        gradient, = torch.autograd.grad(torch.dot(action(v), w), v)
        close(gradient, adjoint(w))
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(v.detach(), w)
            close(forward_ad.unpack_dual(action(dual)).tangent, action(w))


def test_cuda_current_stream_capture_and_device_validation():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        A, Q = graph('cuda', torch.float64)
        with pytest.raises(ValueError, match='dtype/device'):
            NullspaceSystem(operator(A), Q.cpu(), Q)
        system = NullspaceSystem(operator(A), Q, Q)
        v = torch.arange(A.shape[0], device='cuda', dtype=A.dtype)
        out = torch.empty_like(v)
        system.apply(v, out)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=side):
            system.apply(v, out)
        v.mul_(2)
        capture.replay()
        expected = system(v)
    side.synchronize()
    close(out, expected)
