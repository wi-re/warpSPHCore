"""Reverse of the full explicit gradient product."""
import pytest
import torch
import warpSPHCore as sph
from _support.gradient_explicit_contract import case,oracle


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('correction',['plain','volume','crk','renorm','combined','native_gradh'])
def test_full_product_reverse_gradcheck(device,kernel,correction):
    index=list(sph.KernelFunctions).index(kernel)
    options=dict(volume=correction in ('volume','combined','native_gradh'),crk=correction in ('crk','combined','native_gradh'),renorm=correction in ('renorm','combined','native_gradh'),gradh=correction=='native_gradh')
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel,support=list(sph.SupportScheme)[index%6],scheme=list(sph.GradientScheme)[index%4],**options)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('correction',['plain','volume','crk','renorm','combined','native_gradh'])
@pytest.mark.parametrize('field_shape',[(),(5,),(2,3,2)])
def test_scalar_weighted_and_packed_reverse_independent_equation(device,dimension,scheme,correction,field_shape):
    options=dict(volume=correction in ('volume','combined','native_gradh'),crk=correction in ('crk','combined','native_gradh'),renorm=correction in ('renorm','combined','native_gradh'),gradh=correction=='native_gradh')
    q,r,d,p,g,v,states,functions=case(device,dimension,scheme=scheme,field_shape=field_shape,**options)
    for weighted in (False,True):
        actual=functions('native')(*v);expected=oracle(v,d,p,g,**options)
        torch.testing.assert_close(actual,expected,rtol=8e-8,atol=8e-8)
        seed=torch.linspace(-.3,.7,actual.numel(),device=device,dtype=actual.dtype).reshape_as(actual) if weighted else torch.ones_like(actual)
        inputs=[x for x in v if x.requires_grad];a=torch.autograd.grad(actual,inputs,seed,allow_unused=True);b=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for k,(x,y,z) in enumerate(zip(a,b,inputs)):torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,rtol=8e-8,atol=8e-8,msg=lambda msg:f'gradient product role {k}: {msg}')


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_deterministic_reverse_and_topology_lifetime(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset),volume=True,crk=True,renorm=True,field_shape=(2,3,2))
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32);g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_(x.requires_grad)
    v=tuple(strided(x) for x in v);run=functions('native');inputs=[x for x in v if x.requires_grad]
    expected=oracle(v,d,p,g,True,True,True);seed=torch.linspace(-.3,.7,expected.numel(),device=device,dtype=expected.dtype).reshape_as(expected);wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
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
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):torch.autograd.grad(y,v[20],torch.ones_like(y))





@pytest.mark.gpu
@pytest.mark.parametrize('volume,crk',[(False,False),(True,True)])
def test_capture_explicit_product_and_its_reverse(device,volume,crk):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,field_shape=(2,3,2),volume=volume,crk=crk,renorm=crk)
        run=functions('native');inputs=[x for x in v if x.requires_grad];seed=torch.linspace(-.3,.7,48,device=device).reshape(2,2,3,2,2)
        def products():
            y=run(*v);bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            return y,*(torch.zeros_like(x) if z is None else z for x,z in zip(inputs,bars))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[0][:,1].add_(.01);v[2].mul_(1.03);v[10].add_(.02);v[27].mul_(1.07)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-4)
    finally:
        if enabled:gc.enable()




@pytest.mark.parametrize('volume,crk',[(False,False),(True,True)])
def test_three_step_product_feedback_reverse(device,volume,crk):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,),volume=volume,crk=crk,renorm=crk)
    def rollout(evaluate,*values):
        position=values[0]
        for _ in range(3):
            field=evaluate(position,*values[1:])
            position=position+.002*field.sum((-2,-1),keepdim=False)[:,None].tanh()*position.cos()
        return torch.cat((position.flatten(),field.flatten()))
    actual=rollout(functions('native'),*v)
    expected=rollout(lambda *x:oracle(x,d,p,g,volume,crk,crk),*v)
    seed=torch.linspace(-.3,.7,actual.numel(),device=device,dtype=actual.dtype)
    torch.testing.assert_close(actual,expected,rtol=5e-8,atol=5e-8)
    inputs=[x for x in v if x.requires_grad]
    a=torch.autograd.grad(actual,inputs,seed,allow_unused=True);b=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
    for x,y,z in zip(a,b,inputs):torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,rtol=5e-8,atol=5e-8)


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_full_product_and_reverse(device,sizes):
    from dataclasses import replace
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,));n,m=sizes;q,r,d=states(v[:10])
    q=replace(q,**{a:getattr(q,a)[:n] for a in ('positions','supports','masses','kinds','densities')});r=replace(r,**{a:getattr(r,a)[:m] for a in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(n,device=device,dtype=torch.int32),torch.zeros(n,device=device,dtype=torch.int64),n,m)
    y=warpOperationJVP(q,p,d,referenceParticles=r,adjacency=g,queryTangentState=sph.ParticleTangentState(v[20][:n],v[22][:n],None,v[25][:n]),
        referenceTangentState=sph.ParticleTangentState(v[21][:m],v[23][:m],v[24][:m],v[26][:m]),queryValues=v[14][:n],referenceValues=v[10][:m],tangentQueryValues=v[31][:n],tangentReferenceValues=v[27][:m])
    assert y.shape==(n,5,d.dim) and not y.count_nonzero()
    for z in torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True):
        if z is not None:assert not z.count_nonzero()


def test_native_search_no_warp_and_shared_role_aliases(device,monkeypatch):
    from dataclasses import replace
    import builtins
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,),volume=True,crk=True,renorm=True)
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name=='warp' or name.startswith('warp.'):raise AssertionError('native gradient product imported Warp')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    y=functions('native',adjacency=None)(*v);torch.testing.assert_close(y,functions('native')(*v),rtol=8e-8,atol=8e-8)
    bars=torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True)
    assert all(z is None or torch.isfinite(z).all() for z in bars)
    q,r,d=states(v[:10]);self_g=sph.AdjacencyList(None,torch.arange(2,device=device).repeat(2),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device),2,2)
    def run(x,h,m,f,tx,th,tm,tf):
        qi=replace(q,positions=x,supports=h,masses=m,densities=f)
        return warpOperationJVP(qi,p,d,adjacency=self_g,queryTangentState=sph.ParticleTangentState(tx,th,None,tf),
            referenceTangentState=sph.ParticleTangentState(tx*.3,th*-.2,tm,tf),queryValues=f,queryVolumes=m,
            tangentQueryValues=tf,tangentQueryVolumes=tm)
    values=(v[0],v[2],v[7],v[8],v[20],v[22],v[22].detach().clone().requires_grad_(),v[22].detach().clone().requires_grad_())
    assert torch.autograd.gradcheck(run,values,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('dimension',[2,3])
def test_packed_corrected_reverse_actual_gradcheck(device,dimension):
    q,r,d,p,g,v,states,functions=case(device,dimension,field_shape=(2,3,2),volume=True,crk=True,renorm=True,gradh=True,scheme=sph.GradientScheme.Symmetric)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_coefficient_only_reverse_actual_gradcheck(device,kernel):
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,kernel=kernel,volume=True,crk=True,renorm=True,gradh=True,geometry=False,scheme=list(sph.GradientScheme)[i%4])
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,atol=8e-5,rtol=2e-3)
