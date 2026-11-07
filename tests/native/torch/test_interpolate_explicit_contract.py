"""Native full interpolation JVP reverse, including coefficient extensions."""
import pytest
import torch
import warpSPHCore as sph
from _support.interpolate_explicit_contract import case,oracle


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('crk',[False,True])
def test_all_kernel_full_product_reverse_gradcheck(device,kernel,volume,crk):
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,kernel=kernel,support=list(sph.SupportScheme)[i%6],volume=volume,crk=crk)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,atol=6e-5,rtol=1e-3)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('crk',[False,True])
@pytest.mark.parametrize('field_shape',[(),(5,),(2,3,2)])
def test_scalar_weighted_and_packed_reverse_independent_equation(device,dimension,volume,crk,field_shape):
    q,r,d,p,g,v,states,functions=case(device,dimension,volume=volume,crk=crk,field_shape=field_shape)
    for weighted in (False,True):
        actual=functions('native')(*v);expected=oracle(v,d,p,g,volume,crk)
        torch.testing.assert_close(actual,expected,rtol=5e-8,atol=5e-8)
        seed=actual.detach().cos()*.3 if weighted else torch.ones_like(actual)
        inputs=[x for x in v if x.requires_grad];a=torch.autograd.grad(actual,inputs,seed,allow_unused=True);b=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for x,y,z in zip(a,b,inputs):torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,rtol=5e-8,atol=5e-8)


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_deterministic_reverse_and_topology_lifetime(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset),volume=True,crk=True,field_shape=(2,3,2))
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32);g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_(x.requires_grad)
    v=tuple(strided(x) for x in v);run=functions('native');inputs=[x for x in v if x.requires_grad]
    expected=oracle(v,d,p,g,True,True);seed=torch.linspace(-.3,.7,expected.numel(),device=device,dtype=expected.dtype).reshape_as(expected);wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
    prior=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for attempt in range(3):
            y=run(*v);bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            for a,b,x in zip(bars,wanted,inputs):torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=5e-8,atol=5e-8)
            if attempt==0:first=bars
            else:
                for a,b in zip(first,bars):
                    if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(prior)
    y=run(*v);g.paddedJ[2]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):torch.autograd.grad(y,v[14],torch.ones_like(y))



@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_explicit_product_and_reverse(device,sizes):
    from dataclasses import replace
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,));n,m=sizes;q,r,d=states(v[:10])
    q=replace(q,**{a:getattr(q,a)[:n] for a in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{a:getattr(r,a)[:m] for a in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(n,device=device,dtype=torch.int32),torch.zeros(n,device=device,dtype=torch.int64),n,m)
    y=warpOperationJVP(q,p,d,referenceParticles=r,adjacency=g,queryTangentState=sph.ParticleTangentState(v[14][:n],v[16][:n],None,None),
        referenceTangentState=sph.ParticleTangentState(v[15][:m],v[17][:m],v[18][:m],v[19][:m]),referenceValues=v[10][:m],tangentReferenceValues=v[20][:m])
    assert y.shape==(n,5) and not y.count_nonzero()
    for z in torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True):
        if z is not None:assert not z.count_nonzero()


def test_native_search_no_warp_and_alias_reverse(device,monkeypatch):
    from dataclasses import replace
    import builtins
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,),volume=True,crk=True)
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name=='warp' or name.startswith('warp.'):raise AssertionError('native interpolation product imported Warp')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    y=functions('native',adjacency=None)(*v)
    torch.testing.assert_close(y,functions('native')(*v),rtol=5e-8,atol=5e-8)
    bars=torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True)
    assert all(z is None or torch.isfinite(z).all() for z in bars)
    q,r,d=states(v[:10]);self_g=sph.AdjacencyList(None,torch.arange(2,device=device).repeat(2),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device),2,2)
    def run(x,h,m,f,tx,th,tm,tf):
        qi=replace(q,positions=x,supports=h,masses=m,densities=f)
        return warpOperationJVP(qi,p,d,adjacency=self_g,queryTangentState=sph.ParticleTangentState(tx,th,None,None),
            referenceTangentState=sph.ParticleTangentState(tx*.3,th*-.2,tm,tf),queryValues=f,queryVolumes=m,
            tangentQueryValues=tf,tangentQueryVolumes=tm)
    values=(v[0],v[2],v[7],v[8],v[14],v[16],v[16].detach().clone().requires_grad_(),v[16].detach().clone().requires_grad_())
    assert torch.autograd.gradcheck(run,values,fast_mode=True,atol=5e-5,rtol=8e-4)


@pytest.mark.gpu
@pytest.mark.parametrize('volume,crk',[(False,False),(True,True)])
def test_capture_explicit_product_and_its_reverse(device,volume,crk):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,field_shape=(2,3,2),volume=volume,crk=crk)
        run=functions('native');inputs=[x for x in v if x.requires_grad];seed=torch.linspace(-.3,.7,24,device=device).reshape(2,2,3,2)
        def products():
            y=run(*v);bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            return y,*(torch.zeros_like(x) if z is None else z for x,z in zip(inputs,bars))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[0][:,1].add_(.01);v[2].mul_(1.03);v[10].add_(.02);v[20].mul_(1.07)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-4)
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('volume,crk',[(False,False),(True,True)])
def test_three_step_product_feedback_reverse(device,volume,crk):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,),volume=volume,crk=crk)
    def rollout(evaluate,*values):
        position=values[0]
        for _ in range(3):
            field=evaluate(position,*values[1:])
            position=position+.002*field.sum(-1,keepdim=True).tanh()*position.cos()
        return torch.cat((position.flatten(),field.flatten()))
    actual=rollout(functions('native'),*v)
    expected=rollout(lambda *x:oracle(x,d,p,g,volume,crk),*v)
    seed=torch.linspace(-.3,.7,actual.numel(),device=device,dtype=actual.dtype)
    torch.testing.assert_close(actual,expected,rtol=5e-8,atol=5e-8)
    inputs=[x for x in v if x.requires_grad]
    a=torch.autograd.grad(actual,inputs,seed,allow_unused=True);b=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
    for x,y,z in zip(a,b,inputs):torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,rtol=5e-8,atol=5e-8)
