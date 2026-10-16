"""Guarded native CG chunks: replay, failure semantics and bounded cache ownership."""

import pytest
import torch
import warp as wp

from warpSPHCore.linalg import CGWorkspace, Status, cg

requiresCuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def test_capture_requires_cuda():
    with pytest.raises(ValueError, match='CUDA'):
        CGWorkspace(torch.ones(4), 8, capture=True)


@requiresCuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('history', [True, False])
@pytest.mark.parametrize('preconditioner', ['none', 'tensor', 'callable'])
def test_changed_contents_replay_and_owned_results(dtype, history, preconditioner):
    b = torch.linspace(1, 2, 32, dtype=dtype, device='cuda')
    diagonal = torch.linspace(1, 4, 32, dtype=dtype, device='cuda')
    inverse = torch.ones_like(b) * .5
    precond = {'none': None, 'tensor': inverse, 'callable': lambda v: inverse * v}[preconditioner]
    action = lambda v: diagonal * v
    ws = CGWorkspace(b, 128, history=history, capture=True)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    solve = lambda rhs, **kw: cg(action, rhs, precond=precond, workspace=ws, maxiter=128, rtol=tolerance, **kw)
    first = solve(b)
    snapshots = [v.clone() for v in (first.x, first.norms, first.status, first.trueResidual)]
    builds = ws.captureBuilds
    assert builds == 1 and first.setupIterations == 1
    diagonal.mul_(1.5)
    inverse.mul_(.75)
    second = solve(b * 2)
    assert ws.captureBuilds == builds and second.setupIterations == 0
    assert second.statusInt() >= 0
    torch.testing.assert_close(second.x, b * 2 / diagonal, rtol=tolerance * 4, atol=tolerance)
    assert second.matvecs == second.loopIterations + second.verifications
    assert first.matvecs == first.loopIterations + first.verifications + first.setupIterations
    if precond is not None:
        assert first.preconditionerApplications == first.loopIterations + first.restarts + 1 + first.setupIterations
    for before, after in zip(snapshots, (first.x, first.norms, first.status, first.trueResidual)):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    # A new tolerance uses the current target buffer, not a captured scalar.
    loose = cg(action, b, precond=precond, workspace=ws, maxiter=128, rtol=1e-3)
    assert loose.statusInt() >= 0 and loose.trueResidual <= 1e-3 * b.norm()
    assert ws.captureBuilds == builds


@requiresCuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_breakdown_threshold_zero_rhs_and_nonfinite_after_capture(dtype):
    b = torch.ones(16, device='cuda', dtype=dtype)
    scale = torch.full_like(b, 2)
    action = lambda v: scale * v
    ws = CGWorkspace(b, 8, capture=True)
    result = cg(action, b, workspace=ws, maxiter=8, rtol=0)
    assert result.statusInt() == 0 and int(result.iterations) == 1
    assert result.loopIterations == 8
    for value in (-2., float('nan'), float('inf')):
        scale.fill_(value)
        result = cg(action, b, workspace=ws, maxiter=8, rtol=0)
        assert result.statusInt() == Status.INDEFINITE
        assert int(result.iterations) == 0
        assert torch.equal(result.x, torch.zeros_like(b))
    assert ws.captureBuilds == 1
    scale.fill_(1.)
    result = cg(action, b * 0, x0=b, workspace=ws, maxiter=8, rtol=0)
    assert result.statusInt() >= 0 and torch.equal(result.x, b * 0)
    result = cg(action, b, workspace=ws, maxiter=8, threshold=.1)
    assert result.statusInt() == Status.THRESHOLD
    assert int(result.iterations) == 1 and ws.captureBuilds == 2
    torch.testing.assert_close(result.x, b)


@requiresCuda
def test_strict_budget_tail_callback_replacement_and_clear():
    b = torch.ones(32, device='cuda', dtype=torch.float64)
    diagonal = torch.linspace(1, 10, 32, device='cuda', dtype=b.dtype)
    action = lambda v: diagonal * v
    ws = CGWorkspace(b, 32, capture=True)
    result = cg(action, b, workspace=ws, maxiter=5, checkEvery=3, rtol=0)
    assert result.statusInt() == Status.BUDGET
    assert result.loopIterations == int(result.iterations) == 5
    assert result.setupIterations == ws.captureBuilds == 2
    assert result.matvecs == 5 + 2 + 2
    again = cg(action, b * 2, workspace=ws, maxiter=5, checkEvery=3, rtol=0)
    assert again.setupIterations == 0 and ws.captureBuilds == 2
    assert len(ws._graphs) <= 2
    # A callback object replacement invalidates automatically.
    action = lambda v: v * 2
    exact = cg(action, b, workspace=ws, maxiter=8, rtol=0)
    assert exact.statusInt() >= 0 and ws.captureBuilds == 3
    ws.clearCapture()
    assert not ws._graphs
    cg(action, b, workspace=ws, maxiter=8, rtol=0)
    assert ws.captureBuilds == 4
    # Storage replacement hidden in the same closure needs explicit clearing.
    matrix = torch.eye(32, device='cuda', dtype=b.dtype)
    action = lambda v: matrix @ v
    cg(action, b, workspace=ws, maxiter=8, rtol=0)
    matrix = matrix * 4
    ws.clearCapture()
    result = cg(action, b, workspace=ws, maxiter=8, rtol=0)
    torch.testing.assert_close(result.x, b / 4)


@requiresCuda
def test_supplied_output_restarts_history_and_nondefault_stream(monkeypatch):
    b = torch.ones(4, device='cuda')
    action = lambda v: v * 3
    into = lambda v, out: out.copy_(v)  # deliberately inconsistent accelerated action
    ws = CGWorkspace(b, 24, capture=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.cuda._sleep(20_000_000)
        rhs = b * 2
        result = cg(action, rhs, workspace=ws, matvecOut=into, maxiter=24, maxRestarts=2)
        saved = result.x.clone()
    side.synchronize()
    assert result.statusInt() == Status.RESIDUAL_MISMATCH
    assert result.restarts == 2 and result.verifications == 4
    assert result.loopIterations == 24 and result.setupIterations == 1
    torch.testing.assert_close(result.norms[torch.tensor([0, 9, 18], device='cuda')], rhs.norm() * torch.tensor([1., 2., 4.], device='cuda'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **kw: pytest.fail('device-wide synchronization'))
    with torch.cuda.stream(side):
        second = cg(action, rhs * 2, workspace=ws, matvecOut=into, maxiter=24, maxRestarts=2)
    side.synchronize()
    assert second.setupIterations == 0
    torch.testing.assert_close(result.x, saved, rtol=0, atol=0)


@requiresCuda
def test_nested_stream_scopes_preserve_conditional_capture():
    # A temporary wrapper of an already-selected external stream used to
    # unregister its outer wrapper. Conditional capture then said unknown stream.
    from warpSPHCore.autograd.stream import torchWarpStream
    side = torch.cuda.Stream()
    with torch.cuda.stream(side), torchWarpStream(torch.device('cuda')):
        outer = wp.get_stream()
        with torchWarpStream(torch.device('cuda')):
            assert wp.get_stream() is outer
        with wp.ScopedDevice('cpu'):
            with torchWarpStream(torch.device('cuda')):
                assert wp.get_device().is_cuda and wp.get_stream() is outer
        with wp.ScopedCapture() as capture:
            value = wp.zeros(4, device='cuda')
        wp.capture_launch(capture.graph)
    side.synchronize()


@requiresCuda
def test_initial_only_paths_do_not_build_graphs():
    b = torch.ones(2, device='cuda')
    action = lambda v: v
    ws = CGWorkspace(b, 8, capture=True)
    budget = cg(action, b, workspace=ws, maxiter=0)
    assert budget.statusInt() == Status.BUDGET and budget.matvecs == 1
    assert budget.setupIterations == ws.captureBuilds == 0
    exact = cg(action, b, x0=b, workspace=ws, maxiter=8, rtol=0)
    assert exact.statusInt() == 0 and int(exact.iterations) == 0
    assert exact.matvecs == 1 and ws.captureBuilds == 0
    diagonal = torch.tensor([1., 0.], device='cuda')
    x0 = torch.tensor([0., 3.], device='cuda')
    nullspace = cg(lambda v: v * diagonal, b * 0, x0=x0, workspace=ws, maxiter=8, rtol=0)
    assert nullspace.statusInt() == 0 and int(nullspace.iterations) == 0
    torch.testing.assert_close(nullspace.x, x0)
    assert nullspace.x.data_ptr() != x0.data_ptr()
    assert ws.captureBuilds == 0


@requiresCuda
@pytest.mark.filterwarnings('ignore:The CUDA Graph is empty:UserWarning')
def test_setup_avoids_global_sync_and_restores_gc_after_capture_error(monkeypatch):
    import gc
    b = torch.ones(8, device='cuda')
    ws = CGWorkspace(b, 8, capture=True)
    fail = [True]

    def action(v):
        if torch.cuda.is_current_stream_capturing():
            assert not gc.isenabled()
            if fail[0]:
                raise RuntimeError('intentional recording failure')
        return v * 2

    wasEnabled = gc.isenabled()
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **kw: pytest.fail('global synchronization'))
    with pytest.raises(RuntimeError, match='intentional recording failure'):
        cg(action, b, workspace=ws, maxiter=8)
    assert gc.isenabled() == wasEnabled
    assert not ws._busy and not ws._graphs
    fail[0] = False
    result = cg(action, b, workspace=ws, maxiter=8, rtol=0)
    assert result.statusInt() >= 0 and gc.isenabled() == wasEnabled
    torch.testing.assert_close(result.x, b / 2)


@requiresCuda
def test_native_small_matrix_preconditioner_composes_with_capture():
    from warpSPHCore.linalg import applyPinv
    from warpSPHCore.type_config import get_torch_precision
    dtype = get_torch_precision()
    diagonal = torch.tensor([2., 3.], device='cuda', dtype=dtype).repeat(4)
    matrices = torch.diag_embed(diagonal.reshape(4, 2))
    b = torch.ones_like(diagonal)
    action = lambda v: v * diagonal
    precondition = lambda v: applyPinv(v.reshape(4, 1, 2), matrices).reshape(-1)
    ws = CGWorkspace(b, 8, capture=True)
    first = cg(action, b, workspace=ws, precond=precondition, maxiter=8)
    assert first.statusInt() >= 0
    builds = ws.captureBuilds
    matrices.mul_(2)
    diagonal.mul_(2)
    second = cg(action, b * 2, workspace=ws, precond=precondition, maxiter=8)
    assert second.statusInt() >= 0 and ws.captureBuilds == builds
    torch.testing.assert_close(second.x, b * 2 / diagonal)
