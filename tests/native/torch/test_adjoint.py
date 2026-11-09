"""Residual certificates, actual transposes and finite-algorithm semantics."""
import pytest
import torch
from torch.autograd import forward_ad
from warpSPHCore.linalg import AdjointOptions, ResidualActions, implicitSolve, fixedIterations, gmres


def settings(dtype):
    tol=2e-5 if dtype==torch.float32 else 2e-11
    return AdjointOptions(rootAtol=tol,rootRtol=0,linearAtol=tol*.1,linearRtol=tol*.1,maxiter=40,restart=8)


def problem(device,dtype):
    A=torch.tensor([[3.,.4,-.1],[-.7,2.,.3],[.2,-.6,2.5]],device=device,dtype=dtype,requires_grad=True)
    b=torch.tensor([.7,-.4,1.2],device=device,dtype=dtype,requires_grad=True)
    def residual(z,A,b):return A@z+.05*z**3-b
    def primal(z,A,b):
        for _ in range(7):z=z-torch.linalg.solve(A+torch.diag(.15*z*z),residual(z,A,b))
        return z
    return A,b,residual,primal


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_implicit_nonsymmetric_actions_independent_jacobian_and_unroll(device,dtype):
    A,b,residual,primal=problem(device,dtype);start=b.new_zeros(3).requires_grad_();opt=settings(dtype)
    z,report=implicitSolve(residual,start,(A,b),primal,options=opt)
    bar=torch.tensor([.4,-.3,.8],device=device,dtype=dtype)
    grads=torch.autograd.grad(z,(A,b,start),bar)
    J=A.detach()+torch.diag(.15*z.detach()**2);lam=torch.linalg.solve(J.T,bar)
    for a,c in zip(grads,(-lam[:,None]*z.detach()[None,:],lam,torch.zeros_like(start))):torch.testing.assert_close(a,c,atol=opt.rootAtol,rtol=opt.rootAtol)
    unrolled=primal(start,A,b)
    for a,c in zip(grads[:2],torch.autograd.grad(unrolled,(A,b),bar)):torch.testing.assert_close(a,c,atol=opt.rootAtol,rtol=opt.rootAtol)
    actions=ResidualActions(residual,z,(A,b));v=b.detach().cos();w=v.sin()
    torch.testing.assert_close(actions.jvp(v),J@v,atol=opt.rootAtol,rtol=opt.rootAtol)
    torch.testing.assert_close(actions.stateVjp(w),J.T@w,atol=opt.rootAtol,rtol=opt.rootAtol)
    torch.testing.assert_close(w@actions.operator()(v),v@actions.operator().applyAdjoint(w),atol=opt.rootAtol,rtol=opt.rootAtol)
    dA=A.detach()*.01;db=b.detach().cos()*.03
    with forward_ad.dual_level():
        dual,forwardReport=implicitSolve(residual,start,(forward_ad.make_dual(A,dA),forward_ad.make_dual(b,db)),primal,options=opt)
        tangent=forward_ad.unpack_dual(dual).tangent
    expected=torch.linalg.solve(J,db-dA@z.detach())
    torch.testing.assert_close(tangent,expected,atol=opt.rootAtol,rtol=opt.rootAtol)
    assert report.rootResidual<=report.rootThreshold and report.adjoint[-1]['residual']<=report.adjoint[-1]['threshold']
    assert forwardReport.tangent[-1]['residual']<=forwardReport.tangent[-1]['threshold']
    if dtype==torch.float64:
        eps=1e-5
        difference=(primal(start.detach(),A.detach()+eps*dA,b.detach()+eps*db)-primal(start.detach(),A.detach()-eps*dA,b.detach()-eps*db))/(2*eps)
        torch.testing.assert_close(tangent,difference,atol=2e-10,rtol=2e-8)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_aliases_warmstart_finite_iterations_and_control_lifetime(device,dtype):
    theta=torch.tensor([.5,.8],device=device,dtype=dtype,requires_grad=True);initial=torch.ones_like(theta,requires_grad=True)
    residual=lambda z,a,b:2*z-a-b
    primal=lambda z,a,b:(a+b)/2
    z,_=implicitSolve(residual,initial,(theta,theta),primal,options=settings(dtype))
    g,w=torch.autograd.grad(z.sum(),(theta,initial));torch.testing.assert_close(g,torch.ones_like(theta));assert w.count_nonzero()==0
    iteration=lambda z,a,b:.5*z+.25*(a+b)
    finite=fixedIterations(iteration,initial,(theta,theta),3)
    g,w=torch.autograd.grad(finite.sum(),(theta,initial));torch.testing.assert_close(g,theta.new_full(theta.shape,.875));torch.testing.assert_close(w,theta.new_full(theta.shape,.125))
    control=torch.zeros(1,device=device,dtype=torch.int32)
    z,_=implicitSolve(residual,initial,(theta,theta),primal,options=settings(dtype),controls=(control,))
    control.add_(1)
    with pytest.raises(RuntimeError,match='controls changed'):z.sum().backward()
    controlContainer=[0]
    z,_=implicitSolve(residual,initial,(theta,theta),primal,options=settings(dtype),controlToken=lambda:tuple(controlContainer))
    controlContainer[0]=1
    with pytest.raises(RuntimeError,match='controls changed'):z.sum().backward()


def test_unconverged_primal_adjoint_and_budget_are_errors(device):
    A,b,residual,primal=problem(device,torch.float64);start=b.new_zeros(3);opt=settings(b.dtype)
    with pytest.raises(RuntimeError,match='Primal residual is not converged'):
        implicitSolve(residual,start,(A,b),lambda z,*p:z,options=opt)
    z,report=implicitSolve(residual,start,(A,b),primal,options=opt,adjointSolve=lambda action,rhs:torch.zeros_like(rhs))
    with pytest.raises(RuntimeError,match='Adjoint solve did not converge'):z.sum().backward()
    def budget(z,A,b):return gmres(lambda v:A@v,b,maxiter=1,rtol=1e-15,atol=0,restart=1)
    with pytest.raises(RuntimeError,match='status=-14'):
        implicitSolve(lambda z,A,b:A@z-b,start,(A,b),budget,options=AdjointOptions(rootAtol=100,rootRtol=0))


def test_tolerance_sensitivity_and_independent_transpose_solver(device):
    A,b,residual,primal=problem(device,torch.float64);start=b.new_zeros(3);bar=b.detach().sin();offset=b.new_tensor([1.,-.4,.2])
    errors=[]
    for eps in (1e-3,1e-5,1e-7):
        z,report=implicitSolve(residual,start,(A,b),lambda z,*p:primal(z,*p)+eps*offset,options=AdjointOptions(rootAtol=eps*5,rootRtol=0,linearAtol=1e-12,linearRtol=1e-12),adjointSolve=lambda op,rhs:torch.linalg.solve(torch.stack([op(v) for v in torch.eye(3,device=device,dtype=b.dtype)],1),rhs))
        grad=torch.autograd.grad(z,b,bar)[0]
        true=primal(start,A.detach(),b.detach());expected=torch.linalg.solve((A.detach()+torch.diag(.15*true**2)).T,bar)
        errors.append(float((grad-expected).norm()))
        assert report.adjoint[-1]['residual']<1e-12
    assert errors[1]<errors[0]*.02 and errors[2]<errors[1]*.02


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_distinct_nullspaces_gauge_and_compatibility(device,dtype):
    from warpSPHCore.linalg import LinearOperator,NullspaceSystem
    right=torch.tensor([1.,1.,1.],device=device,dtype=dtype);right=right/right.norm()
    left=torch.tensor([1.,-2.,1.],device=device,dtype=dtype);left=left/left.norm()
    eye=torch.eye(3,device=device,dtype=dtype);Pr=eye-right[:,None]*right;Pl=eye-left[:,None]*left
    base=torch.tensor([[3.,.2,-.1],[-.4,2.,.5],[.6,-.2,1.]],device=device,dtype=dtype)
    A=Pl@base@Pr
    operator=LinearOperator((3,3),dtype,torch.device(device),matvec=lambda v:A@v,adjoint=lambda v:A.T@v)
    nullspace=NullspaceSystem(operator,right[:,None],left[:,None])
    b=torch.tensor([.4,-.2,.1],device=device,dtype=dtype,requires_grad=True);scale=b.new_tensor(1.2,requires_grad=True);gauge=b.new_tensor([.03],requires_grad=True)
    with pytest.raises(ValueError,match='incompatible'):
        nullspace.prepareRhs(left)
    rhs=nullspace.projectLeft(b)
    nullspace.prepareRhs(rhs)
    def residual(state,scale,b,gauge):
        z=state[:3];lagrange=state[3:]
        return torch.cat((scale*(A@z)-nullspace.projectLeft(b)+left*lagrange,right[None]@z-gauge))
    def primal(start,scale,b,gauge):
        block=torch.cat((torch.cat((scale*A,left[:,None]),1),torch.cat((right[None],start.new_zeros((1,1))),1)),0)
        return torch.linalg.solve(block,torch.cat((nullspace.projectLeft(b),gauge)))
    z,report=implicitSolve(residual,b.new_zeros(4),(scale,b,gauge),primal,options=settings(dtype),controls=(right,left))
    expected=primal(b.new_zeros(4),scale,b,gauge);seed=z.detach().cos()
    for g,e in zip(torch.autograd.grad(z,(scale,b,gauge),seed),torch.autograd.grad(expected,(scale,b,gauge),seed)):
        torch.testing.assert_close(g,e,atol=settings(dtype).rootAtol,rtol=settings(dtype).rootAtol)
    torch.testing.assert_close(scale*(A@z[:3]),rhs,atol=settings(dtype).rootAtol,rtol=settings(dtype).rootAtol)
    torch.testing.assert_close(right[None]@z[:3],gauge,atol=settings(dtype).rootAtol,rtol=settings(dtype).rootAtol)
    assert abs(float(z[-1].detach()))<settings(dtype).rootAtol


def test_adjoint_tolerance_sensitivity_measures_original_transpose(device):
    dtype=torch.float64;n=20
    A=torch.diag(torch.linspace(1,20,n,device=device,dtype=dtype))+torch.diag(torch.full((n-1,),.7,device=device,dtype=dtype),1)
    b=torch.linspace(-.3,.8,n,device=device,dtype=dtype,requires_grad=True);bar=b.detach().cos();expected=torch.linalg.solve(A.T,bar);errors=[]
    for tolerance in (1e-2,1e-4,1e-8):
        options=AdjointOptions(rootAtol=1e-12,rootRtol=0,linearAtol=0,linearRtol=tolerance,maxiter=200,restart=8)
        z,report=implicitSolve(lambda z,b:A@z-b,torch.zeros_like(b),(b,),lambda start,b:torch.linalg.solve(A,b),options=options)
        g=torch.autograd.grad(z,b,bar)[0];errors.append(float((g-expected).norm()))
        assert report.adjoint[-1]['residual']<=report.adjoint[-1]['threshold']
    assert errors[1]<errors[0]*.03 and errors[2]<errors[1]*.001


def test_implicit_whole_solve_capture_is_explicitly_unsupported(device):
    if str(device)=='cpu':pytest.skip('CUDA capture contract')
    b=torch.ones(2,device=device,requires_grad=True);start=torch.zeros_like(b)
    graph=torch.cuda.CUDAGraph()
    with pytest.raises(NotImplementedError,match='eager convergence'):
        with torch.cuda.graph(graph):
            implicitSolve(lambda z,b:z-b,start,(b,),lambda z,b:b)
