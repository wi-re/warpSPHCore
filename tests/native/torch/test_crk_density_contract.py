"""Ordinary CRK ratio reverse and the native first-order forward extension."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.crk_density_contract import case,equation


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
def test_all_kernel_actual_reverse_and_forward_gradcheck(device,kernel,dimension):
    *_,v,states,functions=case(device,dimension,kernel=kernel)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('volume_mode',['reference','query','none'])
@pytest.mark.parametrize('coefficient_mode',['full','tuple','none'])
def test_scalar_and_weighted_reverse_independent_ratio(device,dimension,volume_mode,coefficient_mode):
    q,r,d,p,g,v,states,functions=case(device,dimension,volume_mode=volume_mode,coefficient_mode=coefficient_mode)
    for weighted in (False,True):
        actual=functions('native')(*v);expected=equation(v,d,p,g,volume_mode,coefficient_mode)
        torch.testing.assert_close(actual,expected,rtol=2e-7,atol=2e-7)
        seed=torch.linspace(-.3,.7,len(actual),device=device,dtype=actual.dtype) if weighted else torch.ones_like(actual)
        bars=torch.autograd.grad(actual,v,seed,allow_unused=True)
        wanted=torch.autograd.grad(expected,v,seed,allow_unused=True)
        for a,b,x in zip(bars,wanted,v):
            torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=2e-7,atol=2e-7)


@pytest.mark.parametrize('role',[2,7,8,9,11,14,15,16,17,18,19])
def test_sole_unread_state_retains_zero_dependency(device,role):
    *_,v,states,functions=case(device,coefficient_mode='tuple')
    v=tuple(x.detach().requires_grad_(i==role) for i,x in enumerate(v));y=functions('native')(*v)
    torch.testing.assert_close(torch.autograd.grad(y.sum(),v[role])[0],torch.zeros_like(v[role]),rtol=0,atol=0)


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_reverse_determinism_and_topology_guard(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset))
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1],device=device,dtype=index)
    g.edgeOffsets=torch.tensor([2,5],device=device,dtype=offset);g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32)
    def strided(x):
        storage=torch.empty((*x.shape,2),device=device,dtype=x.dtype);storage[...,0]=x.detach()
        return storage[...,0].requires_grad_()
    v=tuple(strided(x) for x in v);prior=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for attempt in range(3):
            y=functions('native')(*v);bars=torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True)
            assert all(x is None or x.isfinite().all() for x in bars)
            if attempt==0:first=bars
            else:
                for a,b in zip(first,bars):
                    if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(prior)
    y=functions('native')(*v);g.paddedJ[2]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):
        torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True)


def test_efficient_zero_cotangent_storage(device):
    *_,v,states,functions=case(device);y=functions('native')(*v)
    bars=torch.autograd.grad(y,v,torch._efficientzerotensor(y.shape,device=device,dtype=y.dtype),allow_unused=True)
    assert all(x is None or not x.count_nonzero() for x in bars)


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_populations_and_reverse(device,sizes):
    from warpSPHCore.crk.native.nt_density import computeCRKDensity
    q,r,d,p,g,v,states,_=case(device);q,r,d=states(v[:10]);n,m=sizes
    q=replace(q,**{a:getattr(q,a)[:n] for a in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{a:getattr(r,a)[:m] for a in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(n,device=device,dtype=torch.int32),torch.zeros(n,device=device,dtype=torch.int64),n,m)
    y=computeCRKDensity(q,p,d,sph.CRKState(v[12][:n],v[13][:n]),None,v[10][:m],g,r)
    assert y.shape==(n,) and not y.count_nonzero()
    for bar in torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True):
        if bar is not None:assert not bar.count_nonzero()


def test_native_search_no_warp_and_shared_role_actual_gradcheck(device,monkeypatch):
    import builtins
    from warpSPHCore.crk.native.nt_density import computeCRKDensity
    q,r,d,p,g,v,states,functions=case(device)
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name=='warp' or name.startswith('warp.'):raise AssertionError('Native CRK density imported Warp')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    torch.testing.assert_close(functions('native',adjacency=None)(*v),functions('native')(*v),rtol=2e-7,atol=2e-7)
    q,_,d=states(v[:10]);g=sph.AdjacencyList(None,torch.arange(2,device=device).repeat(2),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device),2,2)
    def run(x,h,m,V,A,B):
        qi=replace(q,positions=x,supports=h,masses=m)
        return computeCRKDensity(qi,p,d,sph.CRKState(A,B),None,V,g)
    assert torch.autograd.gradcheck(run,(v[0],v[2],v[7],v[10][:2],v[12],v[13]),fast_mode=True,check_forward_ad=True,atol=8e-5,rtol=2e-3)


def test_three_step_feedback_reverse(device):
    q,r,d,p,g,v,states,functions=case(device)
    def rollout(evaluate):
        position=v[0]
        for _ in range(3):
            density=evaluate(position,*v[1:]);position=position+.002*density[:,None].tanh()*position.cos()
        return torch.cat((position.flatten(),density))
    actual=rollout(functions('native'));expected=rollout(lambda *x:equation(x,d,p,g))
    seed=torch.linspace(-.3,.7,actual.numel(),device=device,dtype=actual.dtype)
    torch.testing.assert_close(actual,expected,rtol=2e-7,atol=2e-7)
    a=torch.autograd.grad(actual,v,seed,allow_unused=True);b=torch.autograd.grad(expected,v,seed,allow_unused=True)
    for left,right,x in zip(a,b,v):
        torch.testing.assert_close(torch.zeros_like(x) if left is None else left,torch.zeros_like(x) if right is None else right,rtol=2e-7,atol=2e-7)


@pytest.mark.parametrize('volume_mode,coefficient_mode,query_count',
    [(v,c,17) for v in ('reference','query','none') for c in ('full','tuple','none')]
    + [('reference',c,11) for c in ('full','tuple')])
def test_owned_rows_volume_aliases_and_query_only_coefficients(device,volume_mode,coefficient_mode,query_count):
    from test_row_chunks import sample
    from warpSPHCore.crk.native.nt_density import computeCRKDensity
    particles,domain,policy=sample(device,count=17)
    reference=replace(particles,masses=torch.linspace(.3,.9,17,device=device,dtype=particles.positions.dtype))
    query=replace(particles,**{a:getattr(particles,a)[:query_count] for a in ('positions','supports','masses','kinds','densities')})
    query=replace(query,positions=query.positions+.03)
    properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,operationMode=sph.OperationDirection.TrueAllToToAll)
    owned=sph.radiusSearchRowChunks(query,domain,referenceParticles=reference,policy=replace(policy,maxRows=3),mode=sph.SupportScheme.Scatter)
    flat=sph.radiusSearch(query,domain,referenceParticles=reference,mode=sph.SupportScheme.Scatter)
    volume=torch.linspace(.2,1.2,17,device=device,dtype=query.positions.dtype)
    coefs=None if coefficient_mode=='none' else sph.CRKState(torch.ones_like(query.supports),query.positions.sin()*.02)
    if coefficient_mode=='tuple':coefs=(coefs,sph.CRKState(volume,reference.positions.cos()))
    qv=volume if volume_mode=='query' else None;rv=volume if volume_mode=='reference' else None
    actual=computeCRKDensity(query,properties,domain,coefs,qv,rv,owned,reference)
    expected=computeCRKDensity(query,properties,domain,coefs,qv,rv,flat,reference)
    torch.testing.assert_close(actual,expected,rtol=2e-7,atol=2e-7)
    with pytest.raises(NotImplementedError,match='AD'):
        computeCRKDensity(replace(query,positions=query.positions.clone().requires_grad_()),properties,domain,coefs,qv,rv,owned,reference)


@pytest.mark.gpu
@pytest.mark.parametrize('dimension',[1,2,3])
def test_nondefault_stream_capture_primal_reverse_and_native_dual(device,dimension):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        *_,v,states,functions=case(device,dimension,torch.float32);run=functions('native')
        tangents=tuple(x.detach().cos()*.02 for x in v)
        def products():
            y=run(*v);bars=torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True)
            with torch.autograd.forward_ad.dual_level():
                dual=[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,tangents)]
                product=torch.autograd.forward_ad.unpack_dual(run(*dual)).tangent
            return y,*(torch.zeros_like(x) if z is None else z for x,z in zip(v,bars)),product
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[0][:,0].add_(.01);v[3].mul_(1.03);v[10].mul_(1.05)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-4)
    finally:
        if enabled:gc.enable()
