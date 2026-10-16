"""`warpSPHCore.linalg`: the sync-free symmetric pseudo-inverse against `torch.linalg.pinv` (float64 truth), its rtol convention, the
fused `V @ pinv(M)` for rectangular V, the torch fallback for tapes, and shape errors."""

import pytest
import torch
import warp as wp


from warpSPHCore.linalg import applyPinv, pinvSymmetric
from warpSPHCore import get_torch_precision


@pytest.fixture(scope='module', autouse=True)
def _init_warp_once():
    wp.init()


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return request.param


def _matrices(dim, n=6000, seed=0, device='cpu'):
    dt = get_torch_precision()
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(n, dim, dim, device=device, dtype=dt, generator=g)
    M = A @ A.mT                                                    # SPD, random conditioning
    if dim > 1:
        b = torch.randn(n // 4, dim, 1, device=device, dtype=dt, generator=g)
        M[500:500 + n // 4] = b @ b.mT                              # rank one: the null direction must stay dropped
    M[-500:] *= -1                                                  # signed eigenvalues: 1 / lambda, not 1 / |lambda|
    return M


def _relErr(a, truth):
    scale = truth.abs().flatten(1).amax(dim=1).clamp(min=1e-30).view(-1, *[1] * (truth.ndim - 1))
    return ((a.double() - truth).abs() / scale).flatten(1).amax(dim=1)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_pinvSymmetricIsAsAccurateAsTorch(dim, device):
    M = _matrices(dim, device=device)
    truth = torch.linalg.pinv(M.double())
    err = _relErr(pinvSymmetric(M), truth)
    rows = torch.linalg.cond(M.double()) < 1e3                      # both implementations are accurate there; beyond ~1e4 both are noise
    assert rows.sum() > 1000
    assert err[rows].max() < 500 * torch.finfo(M.dtype).eps
    assert torch.isfinite(pinvSymmetric(M)).all()


@pytest.mark.parametrize('dim, p', [(2, 1), (2, 2), (3, 2), (3, 3)])
def test_applyPinvIsVTimesPinvForRectangularV(dim, p, device):
    M = _matrices(dim, device=device)
    g = torch.Generator(device=device).manual_seed(101 + dim + p)
    V = torch.randn(M.shape[0], p, dim, device=device, dtype=M.dtype, generator=g)
    P = torch.linalg.pinv(M.double())
    truth = V.double() @ P
    error = (applyPinv(V, M).double() - truth).abs().flatten(1).amax(1)
    # ||V dP||_max <= ||V||_infinity ||dP||_max. Normalizing by
    # ||V P|| instead can amplify cancellation in the product arbitrarily.
    # The pinv test independently bounds dP by 500 eps; allow the remaining
    # 12 eps for the small matrix product and reference rounding.
    scale = V.double().abs().sum(-1).amax(-1) * P.abs().flatten(1).amax(1)
    rows = torch.linalg.cond(M.double()) < 1e3
    assert (error[rows] < 512 * torch.finfo(M.dtype).eps * scale[rows]).all()


def test_rtolIsTheCutOff(device):
    """diag(1, 1e-5): kept at torch's default (rtol ~ 2.4e-7 in float32), dropped with rtol = 1e-4."""
    dt = get_torch_precision()
    M = torch.diag(torch.tensor([1.0, 1e-5], dtype=dt, device=device))[None]
    assert pinvSymmetric(M)[0, 1, 1].item() == pytest.approx(1e5, rel=1e-3)
    assert pinvSymmetric(M, rtol=1e-4)[0, 1, 1].item() == 0.0


def test_zeroMatrixAndEmptyBatch(device):
    dt = get_torch_precision()
    assert torch.equal(pinvSymmetric(torch.zeros(3, 3, 3, device=device, dtype=dt)), torch.zeros(3, 3, 3, device=device, dtype=dt))
    assert pinvSymmetric(torch.zeros(0, 2, 2, device=device, dtype=dt)).shape == (0, 2, 2)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('apply', [False, True])
def test_selected_nonzero_discarded_eigenvalue_ad(device, dim, apply):
    # A deliberately visible cutoff gap: the ordinary Moore-Penrose backward
    # misses the discarded eigenvalue in retained/discarded rotation terms.
    matrix = torch.diag(torch.tensor(([2., .1] if dim == 2 else [2., 1., .1]),
                                    dtype=torch.float64, device=device))[None].requires_grad_()
    values = torch.arange(dim*2, dtype=matrix.dtype, device=device).reshape(1, 2, dim).cos().requires_grad_()
    if apply:
        f = lambda m, v: applyPinv(v, m, rtol=.1)
        inputs = (matrix, values)
    else:
        f = lambda m: pinvSymmetric(m, rtol=.1)
        inputs = (matrix,)
    assert torch.autograd.gradcheck(f, inputs)
    assert torch.autograd.gradgradcheck(f, inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA sync check')
def test_noHostSync():
    device = 'cuda'
    M = _matrices(3, device=device)
    V = torch.randn(M.shape[0], 2, 3, device=device, dtype=M.dtype)
    pinvSymmetric(M), applyPinv(V, M)                               # (compile outside the check)
    torch.cuda.set_sync_debug_mode('error')
    try:
        pinvSymmetric(M)
        applyPinv(V, M)
    finally:
        torch.cuda.set_sync_debug_mode('default')


def test_tapeInputsUseTheTorchPath(device):
    M = _matrices(2, n=2000, device=device).requires_grad_(True)
    V = torch.randn(2000, 2, 2, device=device, dtype=M.dtype)
    out = applyPinv(V, M)
    assert out.requires_grad
    out.sum().backward()
    assert M.grad is not None and torch.isfinite(M.grad).all()


def test_shapeErrors(device):
    dt = get_torch_precision()
    with pytest.raises(ValueError):
        pinvSymmetric(torch.zeros(4, 4, 4, device=device, dtype=dt))
    with pytest.raises(ValueError):
        applyPinv(torch.zeros(4, 2, 3, device=device, dtype=dt), torch.zeros(4, 2, 2, device=device, dtype=dt))


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_strided_batches_match_dense_reference(device, dim):
    dt = get_torch_precision()
    A = torch.arange(1, 1 + 6 * dim * dim, dtype=dt, device=device).reshape(6, dim, dim)
    M = (A @ A.mT + 2 * torch.eye(dim, dtype=dt, device=device))[::2].mT
    V = torch.arange(1, 1 + 6 * 4 * dim, dtype=dt, device=device).reshape(6, 4, dim)[::2, ::2]
    expected = torch.linalg.pinv(M.double()).to(dt)
    P = pinvSymmetric(M)
    torch.testing.assert_close(P, expected, rtol=2e-3 if dt == torch.float32 else 1e-10, atol=2e-5 if dt == torch.float32 else 1e-12)
    torch.testing.assert_close(applyPinv(V, M), V @ expected, rtol=3e-3 if dt == torch.float32 else 1e-10, atol=1e-3 if dt == torch.float32 else 1e-11)
    assert P.is_contiguous()


def test_forward_ad_preserves_both_matrix_and_value_tangents(device):
    dt = get_torch_precision()
    M = torch.diag(torch.tensor([2., 3.], dtype=dt, device=device))[None]
    V = torch.tensor([[[1., 2.], [3., 4.]]], dtype=dt, device=device)
    dM, dV = torch.ones_like(M), torch.ones_like(V)
    P = torch.linalg.inv(M)
    expectedDP = -P @ dM @ P
    with torch.autograd.forward_ad.dual_level():
        m = torch.autograd.forward_ad.make_dual(M, dM)
        v = torch.autograd.forward_ad.make_dual(V, dV)
        _, dp = torch.autograd.forward_ad.unpack_dual(pinvSymmetric(m))
        _, dout = torch.autograd.forward_ad.unpack_dual(applyPinv(v, m))
        assert dp is not None and dout is not None
        torch.testing.assert_close(dp, expectedDP)
        torch.testing.assert_close(dout, dV @ P + V @ expectedDP)
    # torch.func uses a separate forward-mode transform path.
    _, dp = torch.func.jvp(pinvSymmetric, (M,), (dM,))
    torch.testing.assert_close(dp, expectedDP)


def test_forward_ad_of_values_alone(device):
    dt = get_torch_precision()
    M = torch.eye(2, dtype=dt, device=device)[None]
    V = torch.ones_like(M)
    _, dv = torch.func.jvp(lambda v: applyPinv(v, M), (V,), (V * 2,))
    torch.testing.assert_close(dv, V * 2)


@pytest.mark.parametrize('rtol', [-1, float('nan'), float('inf')])
def test_invalid_tolerance(device, rtol):
    M = torch.eye(2, dtype=get_torch_precision(), device=device)[None]
    with pytest.raises(ValueError, match='rtol'):
        pinvSymmetric(M, rtol=rtol)
    with pytest.raises(ValueError, match='rtol'):
        applyPinv(M, M, rtol=rtol)


def test_native_precision_and_value_dtype_are_explicit(device):
    dt = get_torch_precision()
    other = torch.float64 if dt == torch.float32 else torch.float32
    M = torch.eye(2, dtype=dt, device=device)[None]
    with pytest.raises(ValueError, match='process precision'):
        pinvSymmetric(M.to(other))
    with pytest.raises(ValueError, match='same dtype'):
        applyPinv(M.to(other), M)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream/capture check')
def test_nondefault_stream_and_graph_replay(monkeypatch):
    dt = get_torch_precision()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    launched = []
    original = wp.launch

    def launch(*args, **kw):
        launched.append(kw['stream'].cuda_stream)
        return original(*args, **kw)

    monkeypatch.setattr(wp, 'launch', launch)
    with torch.cuda.stream(side):
        M = torch.eye(2, dtype=dt, device='cuda')[None].repeat(8, 1, 1)
        V = torch.ones_like(M)
        pinvSymmetric(M), applyPinv(V, M)  # warm before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            P, out = pinvSymmetric(M), applyPinv(V, M)
        M.mul_(2)
        graph.replay()
        Pcopy, outcopy = P.clone(), out.clone()
    side.synchronize()
    assert launched and all(stream == side.cuda_stream for stream in launched)
    torch.testing.assert_close(Pcopy, torch.eye(2, dtype=dt, device='cuda')[None].expand_as(Pcopy) / 2)
    torch.testing.assert_close(outcopy, V / 2)
