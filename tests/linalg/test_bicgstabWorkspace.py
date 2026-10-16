"""Native BiCGStab: independent systems, failure exits, ownership and capture."""

import pytest
import torch

from warpSPHCore.linalg import BiCGStabWorkspace, BlockJacobiPreconditioner, Status, bicgstab


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.fixture(params=[False, True], ids=['eager', 'cached'])
def capture(request, device):
    if request.param and device == 'cpu':
        pytest.skip('capture requires CUDA')
    return request.param


def system(device, dtype, n=24):
    generator = torch.Generator().manual_seed(27)
    A = torch.eye(n, dtype=dtype)*2 + .06*torch.randn(n, n, dtype=dtype, generator=generator)
    b = torch.randn(n, dtype=dtype, generator=generator)
    return A.to(device), b.to(device)


@pytest.mark.parametrize('history', [False, True])
@pytest.mark.parametrize('preconditioner', ['none', 'diagonal', 'block_out'])
def test_verified_nonsymmetric_solve_reuse_and_callback_counts(device, dtype, capture, history, preconditioner):
    A, b = system(device, dtype)
    ws = BiCGStabWorkspace(b, 80, history=history, capture=capture)
    ptrs = {k:v.data_ptr() for k,v in ws._t.items()}
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    inverse = A.diag().reciprocal()
    blocks = torch.stack([A[i:i+2, i:i+2] for i in range(0, b.numel(), 2)])
    helper = BlockJacobiPreconditioner(blocks, application='native')
    precond = {'none':None, 'diagonal':inverse, 'block_out':helper}[preconditioner]
    precondOut = helper.apply if preconditioner == 'block_out' else None
    into = lambda v, out: torch.mv(A, v, out=out)
    action = lambda v: A @ v

    def solve(rhs):
        return bicgstab(action, rhs, workspace=ws, matvecOut=into, precond=precond,
                        precondOut=precondOut, maxiter=80, rtol=tolerance)

    first = solve(b)
    saved = [v.clone() for v in (first.x, first.norms, first.valid, first.status, first.iterations, first.trueResidual)]
    second = solve(2*b)
    for result, rhs in ((first,b),(second,2*b)):
        assert result.statusInt() >= 0
        torch.testing.assert_close(result.x, torch.linalg.solve(A.double(),rhs.double()).to(dtype), rtol=10*tolerance, atol=tolerance)
        torch.testing.assert_close(result.trueResidual, (rhs-A@result.x).norm(), rtol=0, atol=0)
        assert result.trueResidual <= tolerance*rhs.norm()
        assert result.matvecs == 2*(result.loopIterations+result.setupIterations)+result.verifications
        assert result.preconditionerApplications == (2*(result.loopIterations+result.setupIterations) if precond is not None else 0)
        assert result.x.data_ptr() != ws._t['x'].data_ptr()
        assert result.norms.numel() > 0 if history else result.norms.numel() == 0
    for before, after in zip(saved,(first.x,first.norms,first.valid,first.status,first.iterations,first.trueResidual)):
        torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert ptrs == {k:v.data_ptr() for k,v in ws._t.items()}
    assert ws.nbytes >= 10*b.numel()*b.element_size()
    if capture:
        assert first.setupIterations == 1 and second.setupIterations == 0
        assert ws.captureBuilds == 1


def test_half_step_freezes_and_preserves_threshold_precedence(device, dtype, capture):
    b = torch.ones(4, device=device, dtype=dtype)
    ws = BiCGStabWorkspace(b,8,capture=capture)
    action = lambda v: 2*v
    result = bicgstab(action,b,workspace=ws,maxiter=8,rtol=0)
    assert result.statusInt() == 0 and int(result.iterations) == 1
    assert result.loopIterations == 8
    assert not result.valid.any()  # half exits do not append a full-step norm
    torch.testing.assert_close(result.x,b/2,rtol=0,atol=0)
    threshold = .5 - (1e-10 if dtype == torch.float64 else 1e-5)
    failure = bicgstab(action,b,workspace=ws,maxiter=8,threshold=threshold)
    assert failure.statusInt() == Status.THRESHOLD and failure.iterations == 1
    torch.testing.assert_close(failure.x,b/2,rtol=0,atol=0)


def test_full_step_threshold_and_bad_omega_keep_contract(device, dtype, capture):
    A = torch.diag(torch.tensor([1.,2.],device=device,dtype=dtype))
    b = torch.ones(2,device=device,dtype=dtype)
    ws = BiCGStabWorkspace(b,8,capture=capture)
    result = bicgstab(lambda v:A@v,b,workspace=ws,maxiter=8,threshold=.1,dim=2)
    assert result.statusInt() == Status.THRESHOLD and result.iterations == 1
    assert result.valid.sum() == 1
    torch.testing.assert_close(result.x,torch.tensor([13/15,7/15],device=device,dtype=dtype),rtol=2e-6,atol=2e-6)
    # A nonsingular upper triangular matrix makes s orthogonal A*s, so omega
    # is zero. The unsolved half-step must not be accepted.
    matrix = torch.tensor([[1.,1.],[1.,0.]],device=device,dtype=dtype)
    rhs = torch.tensor([1.,0.],device=device,dtype=dtype)
    omega = bicgstab(lambda v:matrix@v,rhs,workspace=ws,maxiter=8)
    assert omega.statusInt() == Status.DENOMINATOR_BREAKDOWN
    assert omega.iterations == 0 and torch.equal(omega.x,torch.zeros_like(rhs))
    # General zero operator exercises denominator breakdown without NaN x.
    bad = bicgstab(torch.zeros_like,rhs,workspace=ws,maxiter=8)
    assert bad.statusInt() == Status.DENOMINATOR_BREAKDOWN
    assert bad.iterations == 0 and torch.equal(bad.x,torch.zeros_like(rhs))


def test_strict_budget_initial_paths_and_residual_repair(device, dtype, capture):
    A,b = system(device,dtype)
    ws = BiCGStabWorkspace(b,24,capture=capture)
    budget = bicgstab(lambda v:A@v,b,workspace=ws,maxiter=5,checkEvery=3,rtol=0)
    assert budget.loopIterations == 5 and budget.statusInt() == Status.BUDGET
    if capture:
        assert budget.setupIterations == 2 and len(ws._graphs) == 2
    initial = bicgstab(lambda v:A@v,b,workspace=ws,maxiter=0)
    assert initial.matvecs == initial.verifications == 1 and initial.iterations == 0
    wrong = bicgstab(lambda v:3*v,b,workspace=ws,matvecOut=lambda v,out:out.copy_(v),maxiter=24,maxRestarts=2)
    assert wrong.statusInt() == Status.RESIDUAL_MISMATCH
    assert wrong.restarts == 2 and wrong.verifications == 4 and wrong.loopIterations == 24
    torch.testing.assert_close(wrong.trueResidual,(b-3*wrong.x).norm(),rtol=0,atol=0)


def test_nonfinite_warm_start_and_empty_input(device,dtype,capture):
    A,b = system(device,dtype)
    ws = BiCGStabWorkspace(b,80,capture=capture)
    warm = torch.linalg.solve(A,b)
    initial = bicgstab(lambda v:A@v,b,warm,workspace=ws,maxiter=80,rtol=1e-4)
    assert initial.statusInt() == 0 and initial.matvecs == 1
    assert initial.x.data_ptr() != warm.data_ptr()
    failed = bicgstab(lambda v:v*float('nan'),b,workspace=ws,maxiter=8)
    assert failed.statusInt() < 0 and failed.x.isfinite().all()
    empty = torch.empty(0,dtype=dtype,device=device)
    result = bicgstab(lambda v:v,empty,workspace=BiCGStabWorkspace(empty,0,capture=capture),maxiter=0)
    assert result.statusInt() == 0 and result.trueResidual == 0 and result.iterations == 0


def test_validation_callback_errors_and_autograd_release():
    b = torch.ones(4)
    ws = BiCGStabWorkspace(b,8)
    with pytest.raises(ValueError,match='CUDA'):
        BiCGStabWorkspace(b,8,capture=True)
    with pytest.raises(ValueError,match='capacity'):
        bicgstab(lambda v:v,b,workspace=ws,maxiter=9)
    with pytest.raises(ValueError,match='workspace b'):
        bicgstab(lambda v:v,b.double(),workspace=ws,maxiter=8)
    with pytest.raises(ValueError,match='autograd'):
        bicgstab(lambda v:v,b.clone().requires_grad_(),workspace=ws,maxiter=8)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(b,torch.ones_like(b))
        with pytest.raises(ValueError,match='autograd'):
            bicgstab(lambda v:v,dual,workspace=ws,maxiter=8)
    parameter = torch.ones(4,requires_grad=True)
    for callback in (lambda v,out:parameter*v,lambda v,out:out.copy_(parameter*v)):
        with pytest.raises(ValueError,match='supplied output|autograd'):
            bicgstab(lambda v:v,b,workspace=ws,matvecOut=callback,maxiter=8)
        assert not ws._busy
    def replace(v,out):
        out.set_(torch.zeros(5))
    with pytest.raises(ValueError,match='metadata'):
        bicgstab(lambda v:v,b,workspace=ws,matvecOut=replace,maxiter=8)
    exact = bicgstab(lambda v:v,b,workspace=ws,maxiter=8,rtol=0)
    assert exact.statusInt() >= 0
    with pytest.raises(ValueError,match='requires a preconditioner'):
        bicgstab(lambda v:v,b,workspace=ws,maxiter=8,precondOut=lambda v,out:out.copy_(v))
    with pytest.raises(ValueError,match='require a BiCGStabWorkspace'):
        bicgstab(lambda v:v,b,matvecOut=lambda v,out:out.copy_(v))


@pytest.mark.parametrize('seed',[0,2,8])
def test_ill_conditioned_success_is_independently_verified(device, capture, seed):
    generator = torch.Generator().manual_seed(seed)
    Q,_ = torch.linalg.qr(torch.randn(32,32,dtype=torch.float64,generator=generator))
    A = ((Q*torch.logspace(0,4,32,dtype=torch.float64))@Q.T).float().to(device)
    b = torch.randn(32,generator=generator).to(device)
    ws = BiCGStabWorkspace(b,160,capture=capture)
    result = bicgstab(lambda v:A@v,b,workspace=ws,maxiter=160,rtol=1e-6)
    actual = (b-A@result.x).norm()
    torch.testing.assert_close(result.trueResidual,actual,rtol=0,atol=0)
    assert result.statusInt() < 0 or actual <= 1e-6*b.norm()
    assert result.x.isfinite().all() and result.loopIterations <= 160


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_capture_invalidation_contents_streams_and_failure_recovery(monkeypatch):
    import gc
    b = torch.ones(8,device='cuda',dtype=torch.float64)
    scale = torch.full_like(b,2)
    precond = torch.ones_like(b)
    action = lambda v:scale*v
    ws = BiCGStabWorkspace(b,16,capture=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args,**kw:pytest.fail('device-wide synchronization'))
    with torch.cuda.stream(side):
        first = bicgstab(action,b,workspace=ws,precond=precond,maxiter=8,rtol=0)
        builds = ws.captureBuilds
        scale.mul_(2)
        precond.mul_(.5)
        second = bicgstab(action,b*2,workspace=ws,precond=precond,maxiter=8,rtol=0)
        assert ws.captureBuilds == builds and second.setupIterations == 0
        torch.testing.assert_close(first.x,b/2,rtol=0,atol=0)
        torch.testing.assert_close(second.x,b/2,rtol=0,atol=0)
        # Callback replacement invalidates; hidden replacement requires clear.
        other = lambda v:scale*v
        bicgstab(other,b,workspace=ws,maxiter=8)
        assert ws.captureBuilds == builds+1
        scale = scale*2
        ws.clearCapture()
        result = bicgstab(other,b,workspace=ws,maxiter=8)
        torch.testing.assert_close(result.x,b/8)
        enabled = gc.isenabled()
        def failing(v):
            if torch.cuda.is_current_stream_capturing():
                assert not gc.isenabled()
                raise RuntimeError('intentional recording failure')
            return v*2
        with pytest.raises(RuntimeError,match='intentional recording'):
            bicgstab(failing,b,workspace=ws,maxiter=8)
        assert not ws._busy and gc.isenabled() == enabled
        assert bicgstab(action,b,workspace=ws,maxiter=8).statusInt() >= 0
    side.synchronize()
