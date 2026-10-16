"""Native CG against independent systems, plus workspace reuse/stream contracts."""

import pytest
import torch

from warpSPHCore.linalg import CGWorkspace, Status, cg


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    return request.param


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('history', [True, False])
@pytest.mark.parametrize('preconditioner', ['none', 'diagonal', 'callable'])
def test_spd_and_serial_reuse(device, dtype, history, preconditioner):
    g = torch.Generator().manual_seed(7)
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g, dtype=dtype))
    A = ((Q * torch.linspace(1, 10, 16, dtype=dtype)) @ Q.T).to(device)
    b = torch.linspace(1, 2, 16, device=device, dtype=dtype)
    ws = CGWorkspace(b, 128, history=history)
    inverse = A.diag().reciprocal()
    precond = {'none': None, 'diagonal': inverse, 'callable': lambda v: inverse * v}[preconditioner]
    ptrs = {key: t.data_ptr() for key, t in ws._t.items()}
    tol = 2e-5 if dtype == torch.float32 else 1e-11
    outCalls = []

    def into(v, out):
        outCalls.append(out.data_ptr())
        torch.mv(A, v, out=out)

    def solve(rhs):
        return cg(lambda v: A @ v, rhs, workspace=ws, matvecOut=into,
                  precond=precond, maxiter=128, rtol=tol)

    first = solve(b)
    snapshots = [v.clone() for v in (first.x, first.norms, first.valid, first.status, first.iterations, first.trueResidual)]
    second = solve(b * 2)
    for result, rhs in ((first, b), (second, b * 2)):
        assert result.statusInt() >= 0
        torch.testing.assert_close(result.x, torch.linalg.solve(A, rhs), rtol=tol * 10, atol=tol)
        torch.testing.assert_close(result.trueResidual, (rhs - A @ result.x).norm(), rtol=0, atol=0)
        assert result.trueResidual <= tol * rhs.norm()
        assert result.x.data_ptr() != ws._t['x'].data_ptr()
        assert result.preconditionerApplications == (result.loopIterations + result.restarts + 1 if precond is not None else 0)
        assert result.matvecs == result.loopIterations + result.verifications
        assert result.norms.numel() > 0 if history else result.norms.numel() == 0
    for before, after in zip(snapshots, (first.x, first.norms, first.valid, first.status, first.iterations, first.trueResidual)):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert ptrs == {key: t.data_ptr() for key, t in ws._t.items()}
    assert set(outCalls) == {ws._t['ap'].data_ptr()}
    assert ws.nbytes >= b.numel() * b.element_size() * 7


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_frozen_updates_and_indefinite_curvature(device, dtype):
    b = torch.ones(4, device=device, dtype=dtype)
    ws = CGWorkspace(b, 8)
    result = cg(lambda v: v, b, workspace=ws, maxiter=8, rtol=0)
    assert result.statusInt() == 0 and int(result.iterations) == 1
    assert result.loopIterations == 8 and result.matvecs == 10
    assert result.valid.tolist() == [True, True] + [False] * 7
    torch.testing.assert_close(result.x, b, rtol=0, atol=0)
    failure = cg(lambda v: -v, b, workspace=ws, maxiter=8)
    assert failure.statusInt() == Status.INDEFINITE
    assert int(failure.iterations) == 0
    assert torch.equal(failure.x, torch.zeros_like(b))


def test_workspace_validation_and_autograd():
    b = torch.ones(4)
    ws = CGWorkspace(b, 8)
    for value in (torch.ones(4, dtype=torch.float64), torch.ones(3)):
        with pytest.raises(ValueError, match='workspace b'):
            cg(lambda v: v, value, maxiter=8, workspace=ws)
    with pytest.raises(ValueError, match='capacity'):
        cg(lambda v: v, b, workspace=ws, maxiter=9)
    with pytest.raises(ValueError, match='autograd'):
        cg(lambda v: v, b.requires_grad_(), workspace=ws, maxiter=8)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(b.detach(), torch.ones_like(b))
        with pytest.raises(ValueError, match='autograd'):
            cg(lambda v: v, dual, workspace=ws, maxiter=8)
    parameter = torch.ones(4, requires_grad=True)
    with pytest.raises(ValueError, match='autograd'):
        cg(lambda v: v * parameter, b.detach(), workspace=ws, maxiter=8)
    # Errors release the workspace for reuse.
    result = cg(lambda v: v, b.detach(), workspace=ws, maxiter=8)
    assert result.statusInt() >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nondefault_stream(device, dtype, monkeypatch):
    if device != 'cuda':
        pytest.skip('requires CUDA')
    b = torch.ones(64, device=device, dtype=dtype)
    ws = CGWorkspace(b, 8)
    cg(lambda v: v * 2, b, workspace=ws, maxiter=8)  # compile
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    original = torch.cuda.synchronize
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **kw: pytest.fail('device-wide synchronization'))
    with torch.cuda.stream(side):
        torch.cuda._sleep(20_000_000)
        rhs = b * 6
        result = cg(lambda v: v * 2, rhs, workspace=ws, maxiter=8, rtol=0)
        consumer = result.x * 2
    side.synchronize()
    monkeypatch.setattr(torch.cuda, 'synchronize', original)
    assert result.statusInt() >= 0
    torch.testing.assert_close(consumer, rhs, rtol=0, atol=0)


def test_bounded_verification_and_empty_input():
    b = torch.ones(4)
    ws = CGWorkspace(b, 24)
    # A broken accelerated action must not manufacture successful convergence.
    result = cg(lambda v: v * 3, b, workspace=ws, matvecOut=lambda v, out: out.copy_(v),
                maxiter=24, maxRestarts=2)
    assert result.statusInt() == Status.RESIDUAL_MISMATCH
    assert result.restarts == 2 and result.verifications == 4
    assert result.loopIterations == 24
    torch.testing.assert_close(result.trueResidual, (b - result.x * 3).norm())
    torch.testing.assert_close(result.norms[torch.tensor([0, 9, 18])], b.norm() * torch.tensor([1., 2., 4.]))
    empty = torch.empty(0, dtype=torch.float64)
    result = cg(lambda v: v, empty, workspace=CGWorkspace(empty, 0), maxiter=0)
    assert result.statusInt() == 0 and result.x.numel() == 0
    assert int(result.iterations) == 0 and result.trueResidual == 0


def test_threshold_uses_tensor_precision_and_group_norm():
    b = torch.tensor([1., 0.], dtype=torch.float64)
    # This positive bound rounds to 1 in float32. Native CG must keep it in
    # float64 when deciding whether the exact answer is over the threshold.
    threshold = 1 - 1e-10
    result = cg(lambda v: v, b, workspace=CGWorkspace(b, 8), maxiter=8, threshold=threshold, dim=2)
    assert result.statusInt() == Status.THRESHOLD
    assert int(result.iterations) == 1
