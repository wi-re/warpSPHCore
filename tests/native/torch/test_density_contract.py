"""Density closure: first-order products, ownership and execution contracts."""
from dataclasses import replace
import gc
import pytest
import torch
import warpSPHCore as sph
from _support.density_contract import case,oracle


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_reverse_and_forward_gradcheck(device,kernel):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel,
        support=list(sph.SupportScheme)[index%6])
    run=functions('native')
    assert torch.autograd.gradcheck(run,v,fast_mode=True,check_forward_ad=True,
        atol=4e-5,rtol=5e-4,check_undefined_grad=True)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_joint_products_match_independent_equation(device,kernel):
    index=list(sph.KernelFunctions).index(kernel)
    dtype=torch.float32 if index%2 else torch.float64
    q,r,d,p,g,v,states,functions=case(device,1+index%3,dtype,kernel=kernel,
        support=list(sph.SupportScheme)[index%6])
    run=functions('native');expected=lambda *x:oracle(x,d,p,g)
    y=run(*v);want=expected(*v);cot=torch.cos(y.detach())
    tol=3e-4 if dtype==torch.float32 else 2e-8
    torch.testing.assert_close(y,want,rtol=tol,atol=tol)
    gradients=torch.autograd.grad(y,v,cot,allow_unused=True)
    reference=torch.autograd.grad(want,v,cot,allow_unused=True)
    tangents=tuple(.03*torch.cos(x.detach()) for x in v)
    for got,ref,value in zip(gradients,reference,v):
        torch.testing.assert_close(torch.zeros_like(value) if got is None else got,
                                   torch.zeros_like(value) if ref is None else ref,rtol=tol,atol=tol)
    with torch.autograd.forward_ad.dual_level():
        product=torch.autograd.forward_ad.unpack_dual(run(*[
            torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,tangents)])).tangent
    _,wanted=torch.autograd.functional.jvp(expected,v,tangents)
    torch.testing.assert_close(product,wanted,rtol=tol,atol=tol)
    torch.testing.assert_close((product*cot).sum(),sum((z*t).sum() for z,t in zip(gradients,tangents) if z is not None),rtol=tol,atol=tol)
    run(*v).backward(cot)
    for value,wanted in zip(v,gradients):torch.testing.assert_close(value.grad,wanted,rtol=tol,atol=tol)


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_prefix_duplicates_empty_row_and_noncontiguous_inputs(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset))
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32)
    g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_()
    v=tuple(strided(x) for x in v)
    actual=functions('native')(*v);expected=oracle(v,d,p,g)
    seed=torch.tensor([.7,-.2],device=device,dtype=v[0].dtype)
    got=torch.autograd.grad(actual,v,seed,allow_unused=True)
    want=torch.autograd.grad(expected,v,seed,allow_unused=True)
    for a,b,x in zip(got,want,v):torch.testing.assert_close(a,torch.zeros_like(x) if b is None else b,rtol=2e-8,atol=2e-8)


def test_deterministic_products_aliasing_and_topology_lifetime(device):
    q,r,d,p,g,v,states,functions=case(device)
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        run=functions('native');seed=v[0].new_tensor([.3,-.7])
        first=torch.autograd.grad(run(*v),v,seed)
        for _ in range(3):
            for a,b in zip(first,torch.autograd.grad(run(*v),v,seed)):torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Query/reference aliases receive the sum of their physical roles.
        from warpSPHCore.coreOperations.native.wrapper import computeDensity
        state=states(v)[0]
        alias_graph=sph.AdjacencyList(None,torch.arange(2,device=device).repeat(2),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device),2,2)
        y=computeDensity(state,state,d,p,alias_graph,backend='native')
        y.backward(seed)
        assert all(torch.isfinite(t.grad).all() for t in (v[0],v[2],v[7]))
    finally:torch.use_deterministic_algorithms(previous)
    output=functions('native')(*v)
    g.paddedJ[0]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):
        torch.autograd.grad(output,v,torch.ones_like(output))


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_and_unused_fields_have_zero_products(device,sizes):
    q,r,d,p,g,v,states,functions=case(device)
    nq,nr=sizes
    q,r,d=states(v)
    q=replace(q,**{name:getattr(q,name)[:nq] for name in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{name:getattr(r,name)[:nr] for name in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(nq,device=device,dtype=torch.int32),torch.zeros(nq,device=device,dtype=torch.int64),nq,nr)
    from warpSPHCore.coreOperations.native.wrapper import computeDensity
    y=computeDensity(q,r,d,p,g,backend='native')
    for got in torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True):
        if got is not None:assert torch.count_nonzero(got)==0
    # A sole trainable, unused density still gets the reference's zero gradient.
    unused=r.densities.detach().clone().requires_grad_()
    y=computeDensity(replace(q,positions=q.positions.detach()),replace(r,positions=r.positions.detach(),densities=unused),d,p,g,backend='native')
    torch.testing.assert_close(torch.autograd.grad(y,unused,torch.ones_like(y))[0],torch.zeros_like(unused))


def test_native_dispatch_auto_search_and_first_order_scope(device,monkeypatch):
    q,r,d,p,g,v,states,functions=case(device)
    import builtins
    original=builtins.__import__
    def guarded(name,*a,**k):
        if 'wp_density' in name:pytest.fail('native density imported Warp density')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    run=functions('native',adjacency=None)
    explicit=functions('native')(*v);actual=run(*v)
    torch.testing.assert_close(actual,explicit)
    first=torch.autograd.grad(actual,v[0],torch.ones_like(actual),create_graph=True)[0]
    assert not first.requires_grad
    with torch.autograd.forward_ad.dual_level():
        y=functions('native')(*[torch.autograd.forward_ad.make_dual(x,torch.ones_like(x)) for x in v])
        tangent=torch.autograd.forward_ad.unpack_dual(y).tangent
    with pytest.raises(NotImplementedError,match='first-order'):
        torch.autograd.grad(tangent.sum(),v[0])


@pytest.mark.gpu
def test_current_stream_capture_reverse_and_jvp_replay(device):
    if device!='cuda':pytest.skip('CUDA execution contract')
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32)
        run=functions('native');seed=v[0].new_tensor([.2,-.6])
        directions=tuple(torch.cos(x.detach())*.03 for x in v)
        def products():
            output=run(*v)
            backward=torch.autograd.grad(output,v,seed)
            with torch.autograd.forward_ad.dual_level():
                dual=[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,directions)]
                tangent=torch.autograd.forward_ad.unpack_dual(run(*dual)).tangent
            return output,backward,tangent
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[4].mul_(1.07);v[0][:,1].add_(.01)
        graph.replay()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):expected=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip((captured[0],*captured[1],captured[2]),(expected[0],*expected[1],expected[2])):
            torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-4)
    finally:
        if enabled:gc.enable()


def test_three_step_density_feedback_reverse_and_joint_jvp(device):
    q,r,d,p,g,v,states,functions=case(device)
    def rollout(evaluate,*values):
        position=values[0]
        for _ in range(3):
            rho=evaluate(position,*values[1:])
            position=position+.002*rho[:,None].tanh()*position.cos()
        return torch.cat((position.flatten(),rho))
    native=lambda *values:rollout(functions('native'),*values)
    independent=lambda *values:rollout(lambda *x:oracle(x,d,p,g),*values)
    y=native(*v);expected=independent(*v);seed=y.detach().sin()
    actual_grad=torch.autograd.grad(y,v,seed)
    expected_grad=torch.autograd.grad(expected,v,seed,allow_unused=True)
    for a,b,value in zip(actual_grad,expected_grad,v):
        torch.testing.assert_close(a,torch.zeros_like(value) if b is None else b,rtol=3e-8,atol=3e-8)
    directions=tuple(x.detach().cos()*.03 for x in v)
    with torch.autograd.forward_ad.dual_level():
        dual=[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,directions)]
        product=torch.autograd.forward_ad.unpack_dual(native(*dual)).tangent
    _,wanted=torch.autograd.functional.jvp(independent,v,directions)
    torch.testing.assert_close(product,wanted,rtol=3e-8,atol=3e-8)
