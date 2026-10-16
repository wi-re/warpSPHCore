"""Independent partitioned references for constrained linear systems."""

import pytest
import torch
from torch.autograd import forward_ad

from warpSPHCore.linalg import (
    BlockJacobiPreconditioner, CGWorkspace, DiagonalPreconditioner,
    LinearOperator, RestrictedSystem, bicg, cg, gmres,
)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def operator(A, *, adjoint=True, supplied=False):
    return LinearOperator(
        A.shape, A.dtype, A.device,
        matvec=None if supplied else lambda v: A @ v,
        matvecOut=(lambda v, out: torch.mv(A, v, out=out)) if supplied else None,
        adjoint=(lambda v: A.T @ v) if adjoint else None)


def problem(device, dtype):
    A = torch.diag(torch.arange(3., 9., device=device, dtype=dtype))
    A += torch.diag(torch.full((5,), -.7, device=device, dtype=dtype), 1)
    A += torch.diag(torch.full((5,), .2, device=device, dtype=dtype), -1)
    A[4, 0] = .4
    b = torch.arange(1., 7., device=device, dtype=dtype)
    free = torch.tensor([4, 0, 2], device=device)
    fixed = torch.tensor([1, 3, 5], device=device)
    g = torch.tensor([0., 2., 0., -1., 0., 4.], device=device, dtype=dtype)
    return A, b, free, fixed, g


def close(actual, expected):
    tol = 2e-5 if actual.dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=tol, atol=tol)


@pytest.mark.parametrize('supplied', [False, True])
@pytest.mark.parametrize('indexDtype', [torch.int32, torch.int64])
def test_partitioned_action_lifting_reconstruction_and_adjoint(device, dtype, supplied, indexDtype):
    A, b, free, fixed, g = problem(device, dtype)
    system = RestrictedSystem(operator(A, supplied=supplied), free.to(indexDtype))
    Aff = A[free[:, None], free]
    Afc = A[free[:, None], fixed]
    expectedRhs = b[free] - Afc @ g[fixed]
    reduced = torch.linalg.solve(Aff.double(), expectedRhs.double()).to(dtype)
    close(system(reduced), Aff @ reduced)
    close(system.liftRhs(b, g), expectedRhs)
    close(system.applyAdjoint(reduced), Aff.T @ reduced)
    w = reduced.cos()
    close(torch.dot(system(reduced), w), torch.dot(reduced, system.applyAdjoint(w)))
    x = system.reconstruct(reduced, g)
    close((b - A @ x)[free], torch.zeros_like(reduced))
    torch.testing.assert_close(x[fixed], g[fixed], rtol=0, atol=0)
    # The constrained rows carry reactions, and must not be certified as zero.
    assert (b - A @ x)[fixed].norm() > 1
    assert system.fullShape == A.shape and system.shape == Aff.shape
    assert system.dtype == dtype and system.device == A.device
    assert torch.equal(system.freeIndices, free)
    assert torch.equal(system.constrainedIndices, fixed)
    assert system.asLinearOperator() is system.asLinearOperator()
    assert system.asLinearOperator().adjoint is not None


@pytest.mark.parametrize('method', ['gmres', 'bicg'])
def test_nonsymmetric_solve_and_changed_boundary_values(device, dtype, method):
    A, b, free, fixed, g = problem(device, dtype)
    system = RestrictedSystem(operator(A), free)
    reducedOp = system.asLinearOperator()
    precond = DiagonalPreconditioner(A.diag()[free])
    tol = 2e-5 if dtype == torch.float32 else 1e-11
    for prescribed in (None, g, g * -2):
        rhs = system.liftRhs(b, prescribed)
        kwargs = dict(precond=precond, rtol=tol, maxiter=32, checkEvery=1)
        result = (gmres(reducedOp, rhs, **kwargs) if method == 'gmres' else
                  bicg(reducedOp, reducedOp.applyAdjoint, rhs,
                       precondT=precond.applyAdjoint, **kwargs))
        assert result.statusInt() >= 0
        x = system.reconstruct(result.x, prescribed)
        directRhs = b[free] if prescribed is None else b[free] - A[free[:, None], fixed] @ prescribed[fixed]
        reference = torch.linalg.solve(A[free[:, None], free].double(), directRhs.double())
        close(x[free], reference)
        assert (b - A @ x)[free].norm() <= tol * rhs.norm()
        torch.testing.assert_close(result.trueResidual, (rhs - system(result.x)).norm(), rtol=0, atol=0)
        assert system.asLinearOperator() is reducedOp


@pytest.mark.parametrize('native', [False, True])
def test_particle_major_block_restriction_and_cg(device, dtype, native):
    blocks = torch.tensor([[[3., .5], [.5, 2.]], [[4., .2], [.2, 3.]],
                           [[2., .1], [.1, 5.]]], device=device, dtype=dtype)
    A = torch.block_diag(*blocks.unbind()) + .05 * torch.ones((6, 6), device=device, dtype=dtype)
    # Particle order [2, 0], component order [0, 1] explicitly expanded.
    free = torch.tensor([4, 5, 0, 1], device=device)
    g = torch.tensor([0., 0., 1., -2., 0., 0.], device=device, dtype=dtype)
    b = torch.arange(1., 7., device=device, dtype=dtype)
    system = RestrictedSystem(operator(A), free)
    diagonalBlocks = torch.stack([A[4:6, 4:6], A[:2, :2]])
    precond = BlockJacobiPreconditioner(diagonalBlocks, requireSPD=True)
    rhs = system.liftRhs(b, g)
    tol = 2e-5 if dtype == torch.float32 else 1e-11
    workspace = CGWorkspace(rhs, 32, history=False) if native else None
    result = cg(system, rhs, precond=precond, rtol=tol, maxiter=32, workspace=workspace,
                matvecOut=system.apply if native else None)
    assert result.statusInt() >= 0
    x = system.reconstruct(result.x, g)
    assert (b - A @ x)[free].norm() <= tol * rhs.norm()
    close(x[free], torch.linalg.solve(A[free[:, None], free].double(), rhs.double()))


def test_owned_index_map_outputs_and_live_full_coefficients(device, dtype):
    A, b, free, fixed, g = problem(device, dtype)
    originalFree = free.clone()
    system = RestrictedSystem(operator(A), free)
    y = system.restrict(b)
    oldAction = system(y)
    oldRhs = system.liftRhs(b, g)
    saved = [t.clone() for t in (free, y, oldAction, oldRhs, b, g)]
    free.zero_()
    system.freeIndices.zero_()
    system.constrainedIndices.zero_()
    torch.testing.assert_close(system.restrict(b), y, rtol=0, atol=0)
    system.liftRhs(b * 2, g * 3)
    system.reconstruct(y * 2, g * -1)
    for actual, expected in zip((y, oldAction, oldRhs, b, g), saved[1:]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    A.mul_(2)
    close(system(y), oldAction * 2)
    assert torch.equal(system.freeIndices, originalFree)
    # Two previously returned results have independent storage.
    y.zero_()
    assert system.restrict(b).norm() > 0


def test_supplied_outputs_and_noncontiguous_vectors(device, dtype):
    A, b, free, fixed, g = problem(device, dtype)
    system = RestrictedSystem(operator(A), free)
    storage = torch.empty(12, device=device, dtype=dtype)
    storage[::2] = b
    v = storage[::2]
    y = system.restrict(v)
    fullOut = torch.full_like(storage, float('nan'))[::2]
    smallOut = torch.full((6,), float('nan'), device=device, dtype=dtype)[::2]
    calls = [
        (lambda out: system.restrict(v, out), b[free], smallOut),
        (lambda out: system.prolong(y, out), system.prolong(y), fullOut),
        (lambda out: system.apply(y, out), system(y), smallOut),
        (lambda out: system.liftRhs(v, g, out), system.liftRhs(b, g), smallOut),
        (lambda out: system.reconstruct(y, g, out), system.reconstruct(y, g), fullOut),
    ]
    for call, expected, destination in calls:
        ptr, stride = destination.data_ptr(), destination.stride()
        assert call(destination) is destination
        close(destination, expected)
        assert destination.data_ptr() == ptr and destination.stride() == stride


@pytest.mark.parametrize('kind', ['all', 'none', 'empty'])
def test_degenerate_coordinate_sets(device, dtype, kind):
    n = 0 if kind == 'empty' else 3
    A = torch.eye(n, device=device, dtype=dtype) * 2
    free = torch.arange(n - 1, -1, -1, device=device) if kind == 'all' else torch.empty(0, device=device, dtype=torch.long)
    calls = []

    def action(v):
        calls.append(True)
        return A @ v

    op = LinearOperator(A.shape, dtype, A.device, action, adjoint=action)
    system = RestrictedSystem(op, free)
    b = torch.arange(1., n + 1., device=device, dtype=dtype)
    g = torch.zeros_like(b) if kind == 'all' else b * 3
    rhs = system.liftRhs(b, g)
    result = cg(system, rhs, rtol=0, maxiter=8)
    assert result.statusInt() >= 0 and result.trueResidual == 0
    close(system.applyAdjoint(result.x), rhs)
    close(system.reconstruct(result.x, g), b / 2 if kind == 'all' else g)
    if kind != 'all':
        assert not calls
    else:
        torch.testing.assert_close(system.liftRhs(b), b[free], rtol=0, atol=0)


def test_inverse_restriction_is_explicit_and_not_a_reduced_inverse(device, dtype):
    A = torch.tensor([[2., 1.], [1., 2.]], device=device, dtype=dtype)
    free = torch.tensor([0], device=device)
    system = RestrictedSystem(operator(A), free)
    inv = torch.linalg.inv(A)
    P = system.restrictInverseAction(operator(inv))
    one = torch.ones(1, device=device, dtype=dtype)
    close(P(one), one * (2 / 3))
    close(torch.linalg.solve(A[:1, :1], one), one / 2)
    assert not torch.allclose(P(one), one / 2)
    close(P.applyAdjoint(one), P(one))
    inv.mul_(2)
    close(P(one), one * (4 / 3))
    for bad in (lambda v: v, operator(A[:1, :1]), operator(A.to(torch.float64 if dtype == torch.float32 else torch.float32))):
        with pytest.raises(ValueError, match='full inverse action'):
            system.restrictInverseAction(bad)
    with pytest.raises(ValueError, match='operator input.*shape'):
        cg(system, one, precond=DiagonalPreconditioner(A.diag()))


def test_missing_adjoint_is_never_inferred():
    A = torch.eye(3)
    system = RestrictedSystem(operator(A, adjoint=False), torch.tensor([0, 2]))
    assert system.asLinearOperator().adjoint is None
    with pytest.raises(ValueError, match='no adjoint'):
        system.applyAdjoint(torch.ones(2))


@pytest.mark.parametrize('indices', [[0, 0], [-1], [3]])
def test_reject_duplicate_and_out_of_range_indices(indices):
    with pytest.raises(ValueError, match='duplicates|out-of-range'):
        RestrictedSystem(operator(torch.eye(3)), torch.tensor(indices))


def test_reject_ambiguous_shapes_and_index_types():
    A = torch.eye(3)
    op = operator(A)
    for bad in ([0, 1], torch.tensor([0., 1.]), torch.tensor([True, False]), torch.tensor([[0]])):
        with pytest.raises(ValueError, match='freeIndices'):
            RestrictedSystem(op, bad)
    with pytest.raises(ValueError, match='LinearOperator'):
        RestrictedSystem(lambda v: v, torch.tensor([0]))
    with pytest.raises(ValueError, match='square'):
        RestrictedSystem(operator(A[:2]), torch.tensor([0]))


def test_vector_validation_and_no_input_output_aliases():
    system = RestrictedSystem(operator(torch.eye(3)), torch.tensor([2, 0]))
    b, y = torch.ones(3), torch.ones(2)
    g = torch.tensor([0., 4., 0.])
    for bad in (torch.ones(3, 1), b.double(), b.long(), None):
        for call in (system.restrict, system.liftRhs):
            with pytest.raises(ValueError):
                call(bad)
    for bad in (torch.ones(2, 1), y.double(), y.long()):
        for call in (system, system.prolong, system.reconstruct, system.applyAdjoint):
            with pytest.raises(ValueError):
                call(bad)
    for prescribed in (b, g * float('nan'), g * float('inf'), g[:2], g.double()):
        for call in (lambda: system.liftRhs(b, prescribed), lambda: system.reconstruct(y, prescribed)):
            with pytest.raises(ValueError):
                call()
    aliased = [lambda: system.restrict(b, b[:2]), lambda: system.prolong(b[:2], b),
               lambda: system.apply(y, y), lambda: system.liftRhs(b, g, b[:2]),
               lambda: system.liftRhs(b, g, g[:2]), lambda: system.reconstruct(y, g, g)]
    for call in aliased:
        with pytest.raises(ValueError, match='overlap'):
            call()
    for out in (torch.empty(3), torch.empty(2, dtype=torch.float64)):
        with pytest.raises(ValueError):
            system.apply(y, out)
    torch.testing.assert_close(b, torch.ones(3), rtol=0, atol=0)
    torch.testing.assert_close(g, torch.tensor([0., 4., 0.]), rtol=0, atol=0)


def test_reduced_action_vector_reverse_and_forward_ad(device, dtype):
    A, b, free, fixed, g = problem(device, dtype)
    system = RestrictedSystem(operator(A), free)
    y = b[free].clone().requires_grad_()
    seed = y.detach().sin()
    gradient, = torch.autograd.grad(torch.dot(system(y), seed), y)
    close(gradient, A[free[:, None], free].T @ seed)
    tangent = y.detach().cos()
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(y.detach(), tangent)
        result = forward_ad.unpack_dual(system(dual))
        close(result.tangent, A[free[:, None], free] @ tangent)


def test_cuda_device_validation_and_current_stream():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        A, b, free, fixed, g = problem('cuda', torch.float64)
        with pytest.raises(ValueError, match='freeIndices'):
            RestrictedSystem(operator(A), free.cpu())
        system = RestrictedSystem(operator(A), free)
        for call in (system.restrict, system.liftRhs):
            with pytest.raises(ValueError, match='device'):
                call(b.cpu())
        result = gmres(system, system.liftRhs(b, g), rtol=1e-11, maxiter=24)
        reconstructed = system.reconstruct(result.x, g)
        residual = (b - A @ reconstructed)[free].norm()
    stream.synchronize()
    assert result.statusInt() >= 0 and residual < 1e-10


def test_cuda_reduced_application_replays_changed_vectors():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    A, b, free, fixed, g = problem('cuda', torch.float64)
    system = RestrictedSystem(operator(A), free)
    v, out = b[free].clone(), torch.empty_like(b[free])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        system.apply(v, out)
        with torch.cuda.graph(graph, stream=side):
            system.apply(v, out)
    torch.cuda.current_stream().wait_stream(side)
    v.mul_(2)
    graph.replay()
    close(out, A[free[:, None], free] @ v)
