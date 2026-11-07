"""Polynomial assembly and supported explicit-product reverse contracts."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.polyfit_contract import case,equation


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
def test_all_kernel_actual_gradcheck(device,kernel,rhs,explicit):
    dimension=1+list(sph.KernelFunctions).index(kernel)%3
    *_,v,functions=case(device,dimension,kernel=kernel,rhs=rhs,explicit=explicit,subtract=rhs)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=not explicit,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('unit',[False,True])
@pytest.mark.parametrize('support',list(sph.SupportScheme))
def test_scalar_and_weighted_reverse_independent_equation(device,rhs,explicit,unit,support):
    q,r,d,p,g,e,v,functions=case(device,rhs=rhs,explicit=explicit,unit=unit,support=support,subtract=rhs)
    for weighted in (False,True):
        actual=functions('native')(*v)
        expected=equation(v,d,p,g,e,rhs=rhs,explicit=explicit,unit=unit,subtract=rhs)
        torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
        seed=torch.linspace(-.3,.7,actual.numel(),device=device,dtype=actual.dtype).reshape_as(actual) if weighted else torch.ones_like(actual)
        a=torch.autograd.grad(actual,v,seed,allow_unused=True);b=torch.autograd.grad(expected,v,seed,allow_unused=True)
        for x,y,z in zip(a,b,v):torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,rtol=3e-7,atol=3e-7)


@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('support',list(sph.SupportScheme))
def test_native_classic_tangent_reverse_actual_gradcheck(device,rhs,volume,support):
    from torch.autograd import forward_ad
    *_,v,functions=case(device,rhs=rhs,subtract=rhs,volume=volume,support=support)
    direction=tuple((x.detach().cos()*.02).requires_grad_() for x in v)
    def product(*args):
        with forward_ad.dual_level():
            dual=tuple(forward_ad.make_dual(x,t) for x,t in zip(args[:len(v)],args[len(v):]))
            return forward_ad.unpack_dual(functions('native')(*dual)).tangent
    assert torch.autograd.gradcheck(product,(*v,*direction),fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_reverse_determinism_and_topology_guard(device,rhs,explicit,index,offset):
    q,r,d,p,g,e,v,functions=case(device,rhs=rhs,explicit=explicit,subtract=rhs,graph_types=(index,offset))
    g.paddedJ=g.paddedJ.to(index);g.edgeOffsets=g.edgeOffsets.to(offset)
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
    y=functions('native')(*v);g.paddedJ[0]=int(g.paddedJ[0])^1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):
        torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True)


@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
def test_efficient_zero_seed_and_metadata_mutation(device,rhs,explicit):
    q,r,d,p,g,e,v,functions=case(device,rhs=rhs,explicit=explicit,subtract=rhs)
    y=functions('native')(*v)
    bars=torch.autograd.grad(y,v,torch._efficientzerotensor(y.shape,device=device,dtype=y.dtype),allow_unused=True)
    assert all(x is None or not x.count_nonzero() for x in bars)
    y=functions('native')(*v);e[0,0]+=1
    with pytest.raises(RuntimeError,match='modified by an inplace'):
        torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True)


@pytest.mark.parametrize('rhs',[False,True])
def test_arbitrary_order_and_explicit_volume_independent_equation(device,rhs):
    q,r,d,p,g,e,v,functions=case(device,dimension=3,rhs=rhs,subtract=rhs,order=5,volume=True)
    actual=functions('native')(*v);expected=equation(v,d,p,g,e,rhs=rhs,subtract=rhs,volume=True)
    torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
    seed=actual.detach().sin();a=torch.autograd.grad(actual,v,seed,allow_unused=True);b=torch.autograd.grad(expected,v,seed,allow_unused=True)
    for x,y,z in zip(a,b,v):torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,rtol=3e-7,atol=3e-7)


@pytest.mark.parametrize('rhs',[False,True])
def test_native_search_no_warp_and_feedback_reverse(device,rhs,monkeypatch):
    import builtins
    q,r,d,p,g,e,v,functions=case(device,rhs=rhs,subtract=rhs)
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name=='warp' or name.startswith('warp.'):raise AssertionError('Native polynomial assembly imported Warp')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    torch.testing.assert_close(functions('native',adjacency=None)(*v),functions('native')(*v),rtol=3e-7,atol=3e-7)
    def rollout(run):
        x=v[0]
        for _ in range(3):
            y=run(x,*v[1:]);x=x+.001*y.sum(-1,keepdim=True).tanh()*x.cos()
        return torch.cat((x.flatten(),y.flatten()))
    actual=rollout(functions('native'));expected=rollout(lambda *z:equation(z,d,p,g,e,rhs=rhs,subtract=rhs))
    seed=actual.detach().cos();a=torch.autograd.grad(actual,v,seed,allow_unused=True);b=torch.autograd.grad(expected,v,seed,allow_unused=True)
    torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
    for x,y,z in zip(a,b,v):torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,rtol=3e-7,atol=3e-7)


@pytest.mark.parametrize('constant',[False,True])
def test_full_rank_public_solve_and_geometry_tangent_reverse_gradcheck(device,constant):
    from torch.autograd import forward_ad
    q,r,d,p,g,e,v,functions=case(device,dimension=1,explicit=True,order=1)
    def solve(*v):
        particle=replace(q,positions=v[0],supports=v[1],masses=v[2],densities=v[3])
        fit=sph.PolyFit.build(particle,d,p.kernel,1,constant=constant,adjacency=g)
        return fit.gradient(v[4])
    def tangent(*v):
        with forward_ad.dual_level():
            dual=[forward_ad.make_dual(x,t) for x,t in zip(v[:5],v[5:])]
            return forward_ad.unpack_dual(solve(*dual)).tangent
    assert torch.autograd.gradcheck(solve,v[:5],fast_mode=True,atol=8e-5,rtol=2e-3)
    assert torch.autograd.gradcheck(tangent,v,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True,'classic'])
def test_nondefault_stream_capture_primal_and_reverse(device,dimension,rhs,explicit):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        *_,v,functions=case(device,dimension,torch.float32,rhs=rhs,explicit=explicit is True,subtract=rhs)
        run=functions('native')
        if explicit=='classic':
            from torch.autograd import forward_ad
            ordinary=run;size=len(v);directions=tuple((x.detach().cos()*.02).requires_grad_() for x in v)
            v=(*v,*directions)
            def run(*args):
                with forward_ad.dual_level():
                    dual=[forward_ad.make_dual(x,t) for x,t in zip(args[:size],args[size:])]
                    return forward_ad.unpack_dual(ordinary(*dual)).tangent
        def products():
            y=run(*v);bars=torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True)
            return y,*(torch.zeros_like(x) if b is None else b for x,b in zip(v,bars))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[0][:,0].add_(.01);v[1].mul_(1.01)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-4)
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_populations_and_reverse(device,rhs,sizes):
    from warpSPHCore.polyfit.native.nt_polyfit import _sum
    from _support.density_contract import case as geometry_case
    q,r,d,p,g,base,states,_=geometry_case(device)
    v=tuple(x.detach().requires_grad_() for x in base);q,r,d=states(v);n,m=sizes
    q=replace(q,**{a:getattr(q,a)[:n] for a in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{a:getattr(r,a)[:m] for a in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(n,device=device,dtype=torch.int32),torch.zeros(n,device=device,dtype=torch.int64),n,m)
    e=torch.tensor(sph.monomialExponents(2,2),device=device,dtype=torch.int32)
    field=r.positions.sin() if rhs else None
    y,counts=_sum(q,p,d,e,field,adjacency=g,referenceParticles=r)
    assert y.shape==(n,12 if rhs else 21) and not y.count_nonzero()
    if counts is not None:assert counts.shape==(n,) and not counts.count_nonzero()
    for bar in torch.autograd.grad(y,v,torch.ones_like(y),allow_unused=True):
        if bar is not None:assert not bar.count_nonzero()


@pytest.mark.parametrize('rhs',[False,True])
def test_empty_basis_and_components(device,rhs):
    from warpSPHCore.polyfit.native.nt_polyfit import _sum
    q,r,d,p,g,e,v,_=case(device)
    e=e[:0]
    field=v[-1] if rhs else None
    y,counts=_sum(q,p,d,e,field,adjacency=g,referenceParticles=r)
    assert y.shape==(len(q.positions),0)
    if counts is not None:assert not counts.count_nonzero()
    if rhs:
        e=torch.tensor(sph.monomialExponents(d.dim,2),device=device,dtype=torch.int32)
        y,_=_sum(q,p,d,e,v[-1][:,:0],adjacency=g,referenceParticles=r)
        assert y.shape==(len(q.positions),0)


@pytest.mark.parametrize('role',[7,8,4,9,10])
@pytest.mark.parametrize('rhs',[False,True])
def test_sole_ignored_state_and_volume_fallback(device,role,rhs):
    from warpSPHCore.polyfit.native.nt_polyfit import _sum
    from _support.density_contract import case as geometry_case
    q,r,d,p,g,base,states,_=geometry_case(device)
    volume=torch.linspace(.2,.7,len(r.positions),device=device,dtype=base[0].dtype)
    v=tuple(x.detach().requires_grad_(i==role) for i,x in enumerate((*base,volume)))
    q,r,d=states(v[:10]);e=torch.tensor(sph.monomialExponents(d.dim,2),device=device,dtype=torch.int32)
    field=r.positions.sin() if rhs else None
    # Explicit reference volumes override mass/density and query-volume state.
    fallback=v[10]*.9
    y,_=_sum(q,p,d,e,field,adjacency=g,referenceParticles=r,referenceVolumes=v[10].detach(),queryVolumes=fallback)
    bar=torch.autograd.grad(y.sum(),v[role])[0]
    torch.testing.assert_close(bar,torch.zeros_like(bar),rtol=0,atol=0)
    a,_=_sum(q,p,d,e,field,adjacency=g,referenceParticles=r,queryVolumes=v[10])
    b,_=_sum(q,p,d,e,field,adjacency=g,referenceParticles=r,referenceVolumes=v[10])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
