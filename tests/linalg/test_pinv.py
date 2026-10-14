"""`warpSPHCore.linalg`: the sync-free symmetric pseudo-inverse against `torch.linalg.pinv` (float64 truth), its rtol convention, the
fused `V @ pinv(M)` for rectangular V, the torch fallback for tapes, and shape errors."""

import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='warp kernels')

from warpSPHCore.linalg import applyPinv, pinvSymmetric
from warpSPHCore import get_torch_precision


@pytest.fixture(scope='module', autouse=True)
def _init_warp_once():
    wp.init()


def _matrices(dim, n=6000, seed=0):
    dt = get_torch_precision()
    g = torch.Generator(device='cuda').manual_seed(seed)
    A = torch.randn(n, dim, dim, device='cuda', dtype=dt, generator=g)
    M = A @ A.mT                                                    # SPD, random conditioning
    if dim > 1:
        b = torch.randn(n // 4, dim, 1, device='cuda', dtype=dt, generator=g)
        M[500:500 + n // 4] = b @ b.mT                              # rank one: the null direction must stay dropped
    M[-500:] *= -1                                                  # signed eigenvalues: 1 / lambda, not 1 / |lambda|
    return M


def _relErr(a, truth):
    scale = truth.abs().flatten(1).amax(dim=1).clamp(min=1e-30).view(-1, *[1] * (truth.ndim - 1))
    return ((a.double() - truth).abs() / scale).flatten(1).amax(dim=1)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_pinvSymmetricIsAsAccurateAsTorch(dim):
    M = _matrices(dim)
    truth = torch.linalg.pinv(M.double())
    err = _relErr(pinvSymmetric(M), truth)
    rows = torch.linalg.cond(M.double()) < 1e3                      # both implementations are accurate there; beyond ~1e4 both are noise
    assert rows.sum() > 1000
    assert err[rows].max() < 500 * torch.finfo(M.dtype).eps
    assert torch.isfinite(pinvSymmetric(M)).all()


@pytest.mark.parametrize('dim, p', [(2, 1), (2, 2), (3, 2), (3, 3)])
def test_applyPinvIsVTimesPinvForRectangularV(dim, p):
    M = _matrices(dim)
    V = torch.randn(M.shape[0], p, dim, device='cuda', dtype=M.dtype)
    truth = torch.einsum('ipj,ijk->ipk', V.double(), torch.linalg.pinv(M.double()))
    err = _relErr(applyPinv(V, M), truth)
    rows = torch.linalg.cond(M.double()) < 1e3
    assert err[rows].max() < 500 * torch.finfo(M.dtype).eps


def test_rtolIsTheCutOff():
    """diag(1, 1e-5): kept at torch's default (rtol ~ 2.4e-7 in float32), dropped with rtol = 1e-4."""
    dt = get_torch_precision()
    M = torch.diag(torch.tensor([1.0, 1e-5], dtype=dt, device='cuda'))[None]
    assert pinvSymmetric(M)[0, 1, 1].item() == pytest.approx(1e5, rel=1e-3)
    assert pinvSymmetric(M, rtol=1e-4)[0, 1, 1].item() == 0.0


def test_zeroMatrixAndEmptyBatch():
    dt = get_torch_precision()
    assert torch.equal(pinvSymmetric(torch.zeros(3, 3, 3, device='cuda', dtype=dt)), torch.zeros(3, 3, 3, device='cuda', dtype=dt))
    assert pinvSymmetric(torch.zeros(0, 2, 2, device='cuda', dtype=dt)).shape == (0, 2, 2)


def test_noHostSync():
    M = _matrices(3)
    V = torch.randn(M.shape[0], 2, 3, device='cuda', dtype=M.dtype)
    pinvSymmetric(M), applyPinv(V, M)                               # (compile outside the check)
    torch.cuda.set_sync_debug_mode('error')
    try:
        pinvSymmetric(M)
        applyPinv(V, M)
    finally:
        torch.cuda.set_sync_debug_mode('default')


def test_tapeInputsUseTheTorchPath():
    M = _matrices(2, n=2000).requires_grad_(True)
    V = torch.randn(2000, 2, 2, device='cuda', dtype=M.dtype)
    out = applyPinv(V, M)
    assert out.requires_grad
    out.sum().backward()
    assert M.grad is not None and torch.isfinite(M.grad).all()


def test_shapeErrors():
    dt = get_torch_precision()
    with pytest.raises(ValueError):
        pinvSymmetric(torch.zeros(4, 4, 4, device='cuda', dtype=dt))
    with pytest.raises(ValueError):
        applyPinv(torch.zeros(4, 2, 3, device='cuda', dtype=dt), torch.zeros(4, 2, 2, device='cuda', dtype=dt))
