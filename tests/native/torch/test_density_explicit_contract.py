"""Reverse-over-explicit-JVP qualification for the native density owner."""
import pytest
import torch
import warpSPHCore as sph
from _support.density_explicit_contract import case,oracle


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_explicit_reverse_gradcheck(device,kernel):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel,support=list(sph.SupportScheme)[index%6])
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,atol=5e-5,rtol=8e-4)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',list(sph.SupportScheme))
def test_scalar_and_weighted_product_reverse_independent_equation(device,dimension,support):
    q,r,d,p,g,v,states,functions=case(device,dimension,support=support)
    for weighted in (False,True):
        actual=functions('native')(*v);expected=oracle(v,d,p,g)
        torch.testing.assert_close(actual,expected,rtol=5e-8,atol=5e-8)
        seed=actual.new_tensor([.3,-.7]) if weighted else torch.ones_like(actual)
        inputs=[x for x in v if x.requires_grad]
        got=torch.autograd.grad(actual,inputs,seed,allow_unused=True);want=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for a,b,x in zip(got,want,inputs):torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=5e-8,atol=5e-8)


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_deterministic_reverse_and_topology_lifetime(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset))
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32);g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_(x.requires_grad)
    v=tuple(strided(x) for x in v);run=functions('native');inputs=[x for x in v if x.requires_grad]
    expected=oracle(v,d,p,g);seed=expected.new_tensor([.3,-.7]);wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
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


@pytest.mark.gpu
def test_capture_explicit_product_and_its_reverse(device):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32);run=functions('native');inputs=[x for x in v if x.requires_grad];seed=v[2].new_tensor([.3,-.7])
        def products():
            y=run(*v);bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            return y,*(torch.zeros_like(x) if z is None else z for x,z in zip(inputs,bars))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[0][:,1].add_(.01);v[2].mul_(1.03);v[4].mul_(1.07);v[14].add_(.02)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-4)
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_explicit_product_and_reverse(device,sizes):
    from dataclasses import replace
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device);n,m=sizes
    q,r,d=states(v[:10])
    q=replace(q,**{a:getattr(q,a)[:n] for a in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{a:getattr(r,a)[:m] for a in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(n,device=device,dtype=torch.int32),torch.zeros(n,device=device,dtype=torch.int64),n,m)
    qt=sph.ParticleTangentState(v[10][:n],v[12][:n],None,None);rt=sph.ParticleTangentState(v[11][:m],v[13][:m],v[14][:m],None)
    y=warpOperationJVP(q,p,d,referenceParticles=r,adjacency=g,queryTangentState=qt,referenceTangentState=rt)
    assert y.shape==(n,) and not y.count_nonzero()
    for z in torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True):
        if z is not None:assert not z.count_nonzero()


def test_native_search_no_warp_and_alias_reverse(device,monkeypatch):
    from dataclasses import replace
    import builtins
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device);q,r,d=states(v[:10])
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name=='warp' or name.startswith('warp.'):raise AssertionError('native explicit density imported Warp')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    qt=sph.ParticleTangentState(v[10],v[12],None,None);rt=sph.ParticleTangentState(v[11],v[13],v[14],None)
    y=warpOperationJVP(q,p,d,referenceParticles=r,queryTangentState=qt,referenceTangentState=rt)
    torch.testing.assert_close(y,functions('native')(*v),rtol=5e-8,atol=5e-8)
    bars=torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True)
    assert all(z is None or torch.isfinite(z).all() for z in bars)
    self_g=sph.radiusSearch(replace(q,positions=q.positions.detach(),supports=q.supports.detach()),replace(d,min=d.min.detach(),max=d.max.detach()))
    def run(x,h,m,tx,th,tm):
        qi=replace(q,positions=x,supports=h,masses=m)
        return warpOperationJVP(qi,p,d,adjacency=self_g,queryTangentState=sph.ParticleTangentState(tx,th,None,None),
            referenceTangentState=sph.ParticleTangentState(tx*.3,th*-.2,tm,None))
    values=(v[0],v[2],v[7],v[10],v[12],v[12].detach().clone().requires_grad_())
    assert torch.autograd.gradcheck(run,values,fast_mode=True,atol=5e-5,rtol=8e-4)
