"""Gradient closure: first-order products and execution contracts."""
from dataclasses import replace
import gc
import pytest
import torch
import warpSPHCore as sph
from _support.gradient_contract import case,oracle


@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_reverse_and_forward_gradcheck(device,kernel,corrected):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel,
        support=list(sph.SupportScheme)[index%6],corrected=corrected,field_shape=(1+index%5,),scheme=list(sph.GradientScheme)[index%4])
    run=functions('native')
    assert torch.autograd.gradcheck(run,v,fast_mode=True,check_forward_ad=True,
        atol=4e-5,rtol=5e-4,check_undefined_grad=True)


@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_joint_products_match_independent_equation(device,kernel,corrected):
    index=list(sph.KernelFunctions).index(kernel)
    dtype=torch.float32 if index%2 else torch.float64
    q,r,d,p,g,v,states,functions=case(device,1+index%3,dtype,kernel=kernel,
        support=list(sph.SupportScheme)[index%6],corrected=corrected,field_shape=(1+index%5,),scheme=list(sph.GradientScheme)[index%4])
    run=functions('native');expected=lambda *x:oracle(x,d,p,g,corrected)
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
    scalar=torch.autograd.grad(run(*v).sum(),v,allow_unused=True)
    scalar_ref=torch.autograd.grad(expected(*v).sum(),v,allow_unused=True)
    for a,b,x in zip(scalar,scalar_ref,v):
        torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=tol,atol=tol)
    run(*v).backward(cot)
    for value,wanted in zip(v,gradients):torch.testing.assert_close(value.grad,wanted,rtol=tol,atol=tol)


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_prefix_duplicates_empty_row_noncontiguous_and_deterministic(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset),field_shape=(5,),corrected=True)
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32)
    g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_()
    v=tuple(strided(x) for x in v);seed=v[10].new_tensor([[.3,-.2,.1,-.7,.9],[-.5,.4,.1,.2,.8]])[:,:,None].expand(2,5,2).contiguous()
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        expected=oracle(v,d,p,g,True);wanted=torch.autograd.grad(expected,v,seed,allow_unused=True)
        for _ in range(3):
            actual=functions('native')(*v);got=torch.autograd.grad(actual,v,seed)
            torch.testing.assert_close(actual,expected,rtol=3e-8,atol=3e-8)
            for a,b,x in zip(got,wanted,v):torch.testing.assert_close(a,torch.zeros_like(x) if b is None else b,rtol=3e-8,atol=3e-8)
            if _==0:first=got
            else:
                for a,b in zip(first,got):torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(previous)
    out=functions('native')(*v);g.paddedJ[2]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):
        torch.autograd.grad(out,v,torch.ones_like(out))


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_populations_and_sole_unused_input(device,sizes):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,));q,r,d=states(v[:10]);nq,nr=sizes
    q=replace(q,**{n:getattr(q,n)[:nq] for n in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{n:getattr(r,n)[:nr] for n in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(nq,device=device,dtype=torch.int32),torch.zeros(nq,device=device,dtype=torch.int64),nq,nr)
    output=sphOperation(q,p,d,referenceParticles=r,queryValues=v[14][:nq],referenceValues=v[10][:nr],adjacency=g,backend='native')
    assert output.shape==(nq,5,d.dim)
    for gradient in torch.autograd.grad(output,v,torch.ones_like(output),allow_unused=True):
        if gradient is not None:assert not gradient.count_nonzero()
    unused=q.masses.detach().clone().requires_grad_()
    q=replace(q,positions=q.positions.detach(),supports=q.supports.detach(),densities=q.densities.detach(),masses=unused)
    r=replace(r,positions=r.positions.detach(),supports=r.supports.detach(),densities=r.densities.detach(),masses=r.masses.detach())
    output=sphOperation(q,p,replace(d,min=d.min.detach(),max=d.max.detach()),referenceParticles=r,queryValues=v[14][:nq].detach(),referenceValues=v[10][:nr].detach(),adjacency=g,backend='native')
    torch.testing.assert_close(torch.autograd.grad(output,unused,torch.ones_like(output))[0],torch.zeros_like(unused))


def test_aliases_volume_fallback_and_ignored_bundles(device):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device);q=states(v[:10])[0]
    g=sph.AdjacencyList(None,torch.arange(2,device=device).repeat(2),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device),2,2)
    # Shared population and field/volume aliases add their physical-role bars.
    field=q.densities;volume=q.masses
    def run(*x):
        state=replace(q,positions=x[0],supports=x[1],masses=x[2],densities=x[3])
        return sphOperation(state,p,d,queryValues=x[3],queryVolumes=x[2],adjacency=g,
            backend='native')
    values=(q.positions,q.supports,volume,field)
    assert torch.autograd.gradcheck(run,values,fast_mode=True,check_forward_ad=True)


def test_native_search_and_reference_first_order_boundary(device,monkeypatch):
    import builtins
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,))
    original=builtins.__import__
    def guarded(name,*a,**k):
        if 'wp_gradient' in name:pytest.fail('native gradient imported Warp interpolation')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    v=tuple(torch.stack((x.detach(),x.detach()),-1)[...,0].requires_grad_() for x in v)
    actual=functions('native',adjacency=None)(*v)
    torch.testing.assert_close(actual,functions('native')(*v))
    first=torch.autograd.grad(actual,v[0],torch.ones_like(actual),create_graph=True)[0]
    assert not first.requires_grad
    with torch.autograd.forward_ad.dual_level():
        output=functions('native')(*[torch.autograd.forward_ad.make_dual(x,torch.ones_like(x)) for x in v])
        product=torch.autograd.forward_ad.unpack_dual(output).tangent
    with pytest.raises(NotImplementedError,match='first-order'):torch.autograd.grad(product.sum(),v[0])
    with pytest.raises(RuntimeError,match='setup_context'):torch.func.vjp(lambda x:functions('native')(x,*v[1:]),v[0])


@pytest.mark.gpu
def test_current_stream_capture_reverse_and_jvp_replay(device):
    if device!='cuda':pytest.skip('CUDA execution contract')
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,field_shape=(5,),corrected=True)
        run=functions('native');seed=v[10].new_full((2,5,2),.2);directions=tuple(torch.cos(x.detach())*.03 for x in v)
        def products():
            output=run(*v);reverse=torch.autograd.grad(output,v,seed)
            with torch.autograd.forward_ad.dual_level():
                dual=[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,directions)]
                product=torch.autograd.forward_ad.unpack_dual(run(*dual)).tangent
            return output,reverse,product
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[13].mul_(1.07);v[0][:,1].add_(.01);v[10].add_(.02)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):expected=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip((captured[0],*captured[1],captured[2]),(expected[0],*expected[1],expected[2])):
            torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-4)
    finally:
        if enabled:gc.enable()


def test_three_step_gradient_feedback_products(device):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,),corrected=True)
    def rollout(evaluate,*values):
        position=values[0]
        for _ in range(3):
            field=evaluate(position,*values[1:])
            position=position+.002*field.sum((-1,-2))[:,None].tanh()*position.cos()
        return torch.cat((position.flatten(),field.flatten()))
    native=lambda *values:rollout(functions('native'),*values)
    independent=lambda *values:rollout(lambda *x:oracle(x,d,p,g,True),*values)
    y=native(*v);expected=independent(*v);seed=y.detach().sin()
    actual=torch.autograd.grad(y,v,seed);wanted=torch.autograd.grad(expected,v,seed,allow_unused=True)
    for a,b,x in zip(actual,wanted,v):torch.testing.assert_close(a,torch.zeros_like(x) if b is None else b,rtol=3e-8,atol=3e-8)
    directions=tuple(x.detach().cos()*.03 for x in v)
    with torch.autograd.forward_ad.dual_level():
        product=torch.autograd.forward_ad.unpack_dual(native(*[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,directions)])).tangent
    _,want=torch.autograd.functional.jvp(independent,v,directions)
    torch.testing.assert_close(product,want,rtol=3e-8,atol=3e-8)


@pytest.mark.parametrize('field_shape',[(),(5,),(2,5),(2,3,2)])
def test_explicit_joint_corrected_jvp(device,field_shape):
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,field_shape=field_shape,corrected=True)
    q,r,d=states(v[:10]);t=list(x.detach().cos()*.03 for x in v)
    # The explicit API has no omega tangent; implicit duals above cover it.
    t[17]=torch.zeros_like(t[17]);t[19]=torch.zeros_like(t[19])
    arguments=dict(referenceParticles=r,queryValues=v[14],referenceValues=v[10],adjacency=g,
        tangentQueryValues=t[14],tangentReferenceValues=t[10],referenceVolumes=v[13],tangentReferenceVolumes=t[13],
        queryTangentState=sph.ParticleTangentState(t[0],t[2],t[7],t[8]),referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],t[9]),
        crkState=sph.CRKState(v[11],v[12],v[15],v[16]),crkTangentState=sph.CRKTangentState(t[11],t[12],t[15],t[16]),
        gradHState=sph.GradHState(v[19],v[17]),renormalizationState=v[18],renormalizationTangentState=t[18])
    actual=warpOperationJVP(q,p,d,**arguments)
    _,want=torch.autograd.functional.jvp(lambda *x:oracle(x,d,p,g,True),v,tuple(t))
    torch.testing.assert_close(actual,want,rtol=4e-8,atol=4e-8)
