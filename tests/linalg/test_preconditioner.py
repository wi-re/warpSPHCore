"""Independent inverse, transpose, policy, ownership and solver integration gates."""

import pytest
import torch
from torch.autograd import forward_ad

from warpSPHCore.linalg import (
    BlockJacobiPreconditioner, CGWorkspace, DiagonalPreconditioner, bicg,
    bicgstab, cg, gmres,
)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def blocks(device, dtype, d=3, *, spd=False):
    g = torch.Generator().manual_seed(23)
    raw = torch.randn(7, d, d, dtype=torch.float64, generator=g)
    result = raw @ raw.transpose(-1, -2) if spd else raw * .1
    return (result + torch.eye(d) * 2).to(device=device, dtype=dtype)


def close(actual, expected):
    tol = 3e-5 if actual.dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=tol, atol=tol)


@pytest.mark.parametrize('d', [1, 2, 3, 5])
@pytest.mark.parametrize('spd', [False, True])
def test_block_inverse_and_independent_adjoint(device, dtype, d, spd):
    matrix = blocks(device, dtype, d, spd=spd)
    precond = BlockJacobiPreconditioner(matrix, requireSPD=spd)
    v = torch.arange(1, matrix.shape[0] * d + 1, dtype=dtype, device=device)
    expected = torch.linalg.solve(matrix.double(), v.reshape(-1, d).double()).flatten()
    close(precond(v), expected)
    transpose = torch.linalg.solve(matrix.double().transpose(-1, -2), v.reshape(-1, d).double()).flatten()
    close(precond.applyAdjoint(v), transpose)
    w = v.sin()
    close(torch.dot(precond(v), w), torch.dot(v, precond.applyAdjoint(w)))
    assert precond.shape == (v.numel(), v.numel())
    assert precond.spdVerified == spd and precond.fallbackCount == 0
    assert precond.nbytes == matrix.numel() * matrix.element_size()
    assert precond.dtype == dtype and precond.device == matrix.device
    assert precond.asLinearOperator() is precond.asLinearOperator()


def test_diagonal_scale_sign_and_one_dimensional_blocks(device, dtype):
    diagonal = torch.tensor([1e-12, 1e12, -3., 2.], device=device, dtype=dtype)
    precond = DiagonalPreconditioner(diagonal)
    v = torch.ones_like(diagonal)
    torch.testing.assert_close(precond(v), diagonal.reciprocal(), rtol=0, atol=0)
    one = BlockJacobiPreconditioner(diagonal[:, None, None])
    torch.testing.assert_close(one(v), precond(v), rtol=0, atol=0)
    torch.testing.assert_close(precond.applyAdjoint(v), precond(v), rtol=0, atol=0)
    with pytest.raises(ValueError, match='positive definite'):
        DiagonalPreconditioner(diagonal, requireSPD=True)


@pytest.mark.parametrize('block', [False, True])
def test_explicit_cutoff_fallback_shift_and_spd(device, dtype, block):
    cls = BlockJacobiPreconditioner if block else DiagonalPreconditioner
    diagonal = torch.tensor([0., 1e-10, -2., 3.], device=device, dtype=dtype)
    coefficients = torch.diag_embed(diagonal[None]) if block else diagonal
    with pytest.raises(ValueError, match='unusable'):
        cls(coefficients, atol=1e-8)
    pseudo = cls(coefficients, atol=1e-8, singular='pinv')
    v = torch.ones(4, device=device, dtype=dtype)
    close(pseudo(v), torch.tensor([0., 0., -.5, 1/3], device=device, dtype=dtype))
    assert pseudo.fallbackCount == (1 if block else 2)
    identity = cls(coefficients, atol=1e-8, singular='identity')
    expected = torch.ones_like(v) if block else torch.tensor([1., 1., -.5, 1/3], device=device, dtype=dtype)
    close(identity(v), expected)
    shifted = cls(coefficients, shift=4, requireSPD=True)
    close(shifted(v), 1 / (diagonal + 4))
    assert shifted.spdVerified
    zero = torch.zeros_like(coefficients)
    fallback = cls(zero, singular='identity', requireSPD=True)
    close(fallback(v), v)
    with pytest.raises(ValueError, match='rank-deficient|positive definite'):
        cls(zero, singular='pinv', requireSPD=True)


def test_general_pseudoinverse_uses_svd_not_symmetric_part(device, dtype):
    matrix = torch.tensor([[[1., 2.], [0., 0.]], [[1., 0.], [0., 4.]]], dtype=dtype, device=device)
    precond = BlockJacobiPreconditioner(matrix, singular='pinv')
    close(precond.inverse, torch.linalg.pinv(matrix.double()))
    assert precond.fallbackCount == 1
    with pytest.raises(ValueError, match='exactly symmetric'):
        BlockJacobiPreconditioner(matrix, requireSPD=True, singular='identity')


def test_block_relative_cutoff_is_local_to_each_block(device, dtype):
    diagonal = torch.tensor([[1., 1e-4], [1e12, 1e8]], dtype=dtype, device=device)
    precond = BlockJacobiPreconditioner(torch.diag_embed(diagonal), rtol=1e-3, singular='pinv')
    expected = torch.diag_embed(torch.tensor([[1., 0.], [1e-12, 0.]], dtype=dtype, device=device))
    close(precond.inverse, expected)
    assert precond.fallbackCount == 2


@pytest.mark.parametrize('block', [False, True])
def test_ownership_atomic_update_and_storage_stability(device, dtype, block):
    cls = BlockJacobiPreconditioner if block else DiagonalPreconditioner
    coefficients = blocks(device, dtype) if block else torch.arange(1, 8, dtype=dtype, device=device)
    original = coefficients.clone()
    precond = cls(coefficients)
    v = torch.ones(precond.shape[0], dtype=dtype, device=device)
    saved = precond(v)
    ptr = precond._inverse.data_ptr()
    adapter = precond.asLinearOperator()
    coefficients.zero_()
    snapshot = precond.inverse
    snapshot.zero_()
    close(precond(v), saved)
    assert precond.update(original * 2) is precond
    close(precond(v), saved / 2)
    close(adapter(v), saved / 2)
    close(saved, cls(original)(v))
    assert precond.generation == 1 and precond._inverse.data_ptr() == ptr
    for bad in (torch.zeros_like(original), original * float('nan')):
        with pytest.raises(ValueError):
            precond.update(bad)
        close(precond(v), saved / 2)
        assert precond.generation == 1 and precond.fallbackCount == 0
    with pytest.raises(ValueError, match='shape, dtype and device'):
        precond.update(original.to(torch.float64 if dtype == torch.float32 else torch.float32))


@pytest.mark.parametrize('block', [False, True])
def test_strides_supplied_outputs_and_empty_batches(device, dtype, block):
    cls = BlockJacobiPreconditioner if block else DiagonalPreconditioner
    if block:
        matrix = blocks(device, dtype)
        storage = torch.zeros(7, 6, 6, device=device, dtype=dtype)
        coefficients = storage[:, ::2, ::2]
        coefficients.copy_(matrix)
    else:
        storage = torch.arange(1, 15, device=device, dtype=dtype)
        coefficients = storage[::2]
    precond = cls(coefficients)
    n = precond.shape[0]
    v = torch.arange(1, 2*n+1, dtype=dtype, device=device)[::2]
    out = torch.full((2*n,), float('nan'), dtype=dtype, device=device)[::2]
    assert precond.apply(v, out) is out
    close(out, precond(v))
    assert precond.applyAdjoint(v, out) is out
    close(out, precond.applyAdjoint(v))
    with pytest.raises(ValueError, match='overlap'):
        precond.apply(v, v)
    empty = cls(torch.empty((0, 3, 3) if block else (0,), device=device, dtype=dtype))
    assert empty(torch.empty(0, device=device, dtype=dtype)).numel() == 0
    assert empty.nbytes == empty.fallbackCount == 0


@pytest.mark.parametrize('method', ['cg', 'cg_native', 'bicg', 'bicgstab', 'gmres_left', 'gmres_right'])
@pytest.mark.parametrize('application', ['torch', 'native'])
def test_solver_integration_and_true_residual(device, dtype, method, application):
    symmetric = method in ('cg', 'cg_native')
    matrix = blocks(device, dtype, spd=symmetric)
    A = torch.block_diag(*matrix.unbind())
    A = A + .01 * torch.ones_like(A)
    b = torch.arange(1, A.shape[0]+1, dtype=dtype, device=device).sin()
    precond = BlockJacobiPreconditioner(matrix, requireSPD=symmetric, application=application)
    tol = 2e-5 if dtype == torch.float32 else 1e-10
    kw = dict(precond=precond, rtol=tol, maxiter=80)
    if method == 'bicg':
        result = bicg(lambda v: A @ v, lambda v: A.T @ v, b, precondT=precond.applyAdjoint, **kw)
    elif method.startswith('gmres'):
        result = gmres(lambda v: A @ v, b, preconditionerSide=method.split('_')[1], **kw)
    elif method == 'cg_native':
        result = cg(lambda v: A @ v, b, workspace=CGWorkspace(b, 80, history=False), **kw)
    else:
        result = {'cg': cg, 'bicgstab': bicgstab}[method](lambda v: A @ v, b, **kw)
    assert result.statusInt() >= 0
    actual = (b - A @ result.x).norm()
    assert actual <= tol * b.norm()
    torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
    close(result.x, torch.linalg.solve(A.double(), b.double()))


@pytest.mark.parametrize('cls', [DiagonalPreconditioner, BlockJacobiPreconditioner])
@pytest.mark.parametrize('kw', [{'atol': -1}, {'rtol': float('nan')}, {'shift': float('inf')},
                               {'singular': 'auto'}, {'requireSPD': 1}, {'application': 'invalid'}])
def test_bad_policies(cls, kw):
    values = torch.ones(2) if cls is DiagonalPreconditioner else torch.eye(2)[None]
    with pytest.raises(ValueError):
        cls(values, **kw)


@pytest.mark.parametrize('cls', [DiagonalPreconditioner, BlockJacobiPreconditioner])
def test_setup_rejects_ambiguous_shapes_types_and_coefficient_ad(cls):
    value = torch.ones(2) if cls is DiagonalPreconditioner else torch.eye(2)[None]
    for bad in (None, value.long(), value.half(), value.unsqueeze(-1), value.clone().requires_grad_()):
        with pytest.raises(ValueError):
            cls(bad)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(value, torch.ones_like(value))
        with pytest.raises(ValueError, match='differentiate coefficients'):
            cls(dual)
    precond = cls(value)
    with pytest.raises(ValueError, match='operator input'):
        precond(torch.ones(3))


@pytest.mark.parametrize('block', [False, True])
def test_application_vector_derivatives(device, dtype, block):
    cls = BlockJacobiPreconditioner if block else DiagonalPreconditioner
    coefficients = blocks(device, dtype) if block else torch.tensor([2., 3.], dtype=dtype, device=device)
    precond = cls(coefficients, **({'application': 'native'} if block else {}))
    v = torch.ones(precond.shape[0], dtype=dtype, device=device, requires_grad=True)
    gradient, = torch.autograd.grad(precond(v).sum(), v)
    close(gradient, precond.applyAdjoint(torch.ones_like(v)))
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(v.detach(), torch.ones_like(v))
        _, tangent = forward_ad.unpack_dual(precond(dual))
        close(tangent, precond(torch.ones_like(v)))


@pytest.mark.parametrize('kind', ['diagonal', 'block', 'block_native'])
def test_cuda_capture_updates_and_current_stream(kind):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    block = kind != 'diagonal'
    cls = BlockJacobiPreconditioner if block else DiagonalPreconditioner
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        coefficients = blocks('cuda', torch.float64, spd=True) if block else torch.tensor([2., 3.], device='cuda', dtype=torch.float64)
        precond = cls(coefficients, requireSPD=True, application='native' if kind == 'block_native' else 'torch')
        v = torch.ones(precond.shape[0], device='cuda', dtype=torch.float64)
        out = torch.empty_like(v)
        precond.apply(v, out)  # warm libraries
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            precond.apply(v, out)
        graph.replay()
        close(out, precond(v))
        precond.update(coefficients * 2)
        graph.replay()
        close(out, precond(v))
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            with pytest.raises(RuntimeError, match='outside CUDA capture'):
                precond.update(coefficients)


@pytest.mark.parametrize('application', ['torch', 'native'])
def test_cached_cg_sees_same_address_preconditioner_update(application):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    A = torch.tensor([[3., 1.], [1., 2.]], device='cuda', dtype=torch.float64)
    b = torch.ones(2, device='cuda', dtype=A.dtype)
    precond = BlockJacobiPreconditioner(A[None], requireSPD=True, application=application)
    apply = lambda v: A @ v
    workspace = CGWorkspace(b, 16, history=False, capture=True)
    first = cg(apply, b, precond=precond, workspace=workspace, maxiter=16, rtol=1e-10)
    builds = workspace.captureBuilds
    saved = first.x.clone()
    precond.update(A[None] * 2)
    second = cg(apply, b * 2, precond=precond, workspace=workspace, maxiter=16, rtol=1e-10)
    assert first.statusInt() >= 0 and second.statusInt() >= 0
    assert workspace.captureBuilds == builds
    close(first.x, saved)
    close(second.x, 2 * saved)


@pytest.mark.parametrize('d', [1, 2, 3])
def test_native_block_forward_adjoint_strides_and_capture(device, dtype, d):
    matrix = blocks(device, dtype, d)
    precond = BlockJacobiPreconditioner(matrix, application='native')
    n = precond.shape[0]
    v = torch.linspace(-1, 1, n*2, dtype=dtype, device=device)[::2]
    out = torch.empty(n*2, dtype=dtype, device=device)[::2]
    expected = torch.linalg.solve(matrix.double(), v.reshape(-1, d).double()).flatten()
    close(precond(v), expected)
    precond.apply(v, out)
    close(out, expected)
    precond.applyAdjoint(v, out)
    close(out, torch.linalg.solve(matrix.double().transpose(-1, -2), v.reshape(-1, d).double()).flatten())
    if device == 'cuda':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            precond.apply(v, out)
        graph.replay()
        close(out, expected)
        precond.update(matrix * 2)
        graph.replay()
        close(out, expected/2)
    empty = BlockJacobiPreconditioner(matrix[:0], application='native')
    assert empty(v[:0]).numel() == 0


def test_unrepresentable_inverse_and_policy_boundaries():
    tiny = torch.tensor([torch.finfo(torch.float32).smallest_normal / 128])
    with pytest.raises(ValueError, match='unusable'):
        DiagonalPreconditioner(tiny)
    close(DiagonalPreconditioner(tiny, singular='identity')(torch.ones(1)), torch.ones(1))
    with pytest.raises(ValueError, match='1x1..3x3'):
        DiagonalPreconditioner(torch.ones(2), application='native')
    with pytest.raises(ValueError, match='1x1..3x3'):
        BlockJacobiPreconditioner(torch.eye(4)[None], application='native')
