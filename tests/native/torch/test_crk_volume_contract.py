"""Volume's complete native first-order geometry and execution contract."""
from dataclasses import replace
import gc
import pytest
import torch
import warpSPHCore as sph
from _support.crk_volume_contract import case,oracle
from test_curl_contract import check_products


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_reverse_forward_gradcheck(device,kernel):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel,support=list(sph.SupportScheme)[index%6])
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=True,atol=5e-5,rtol=8e-4)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',list(sph.SupportScheme))
def test_joint_products_independent_equation(device,dimension,support):
    q,r,d,p,g,v,states,functions=case(device,dimension,support=support)
    check_products(v,functions('native'),lambda *x:oracle(x,d,p,g),5e-8)


@pytest.mark.parametrize('role',[4,7,8,9])
def test_sole_unread_mass_density_zero_products(device,role):
    q,r,d,p,g,v,states,functions=case(device);v=[x.detach() for x in v]
    v[role]=v[role].requires_grad_();actual=functions('native')(*v)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),v[role])[0],torch.zeros_like(v[role]))
    with torch.autograd.forward_ad.dual_level():
        v[role]=torch.autograd.forward_ad.make_dual(v[role].detach(),torch.ones_like(v[role]))
        product=torch.autograd.forward_ad.unpack_dual(functions('native')(*v)).tangent
    torch.testing.assert_close(product,torch.zeros_like(actual))


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_alias_determinism_and_mutation(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset))
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32);g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_()
    v=tuple(strided(x) for x in v);run=functions('native')
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for attempt in range(3):
            check_products(v,run,lambda *x:oracle(x,d,p,g),5e-8)
            y=run(*v);bars=torch.autograd.grad(y,v,torch.ones_like(y))
            if attempt==0:first=bars
            else:
                for a,b in zip(first,bars):torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(previous)
    y=run(*v);g.paddedJ[2]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):torch.autograd.grad(y,v,torch.ones_like(y))
    # A shared population accumulates query and reference roles into one leaf.
    q,r,d=states(v);self_g=sph.AdjacencyList(None,torch.tensor([0,1,0,1],device=device,dtype=index),
        torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device,dtype=offset),2,2)
    from warpSPHCore.crk.native.nt_volume import computeCRKVolume
    assert torch.autograd.gradcheck(lambda x,h:computeCRKVolume(replace(q,positions=x,supports=h),p,d,self_g),
                                   (v[0],v[2]),fast_mode=True,check_forward_ad=True)


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_populations(device,sizes):
    q,r,d,p,g,v,states,functions=case(device);q,r,d=states(v);n,m=sizes
    q=replace(q,**{a:getattr(q,a)[:n] for a in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{a:getattr(r,a)[:m] for a in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(n,device=device,dtype=torch.int32),torch.zeros(n,device=device,dtype=torch.int64),n,m)
    from warpSPHCore.crk.native.nt_volume import computeCRKVolume
    y=computeCRKVolume(q,p,d,g,r);assert y.shape==(n,) and not y.count_nonzero()
    for value in torch.autograd.grad(y,v,torch.ones_like(y)):assert not value.count_nonzero()


def test_native_search_and_first_order_guards(device,monkeypatch):
    import builtins
    q,r,d,p,g,v,states,functions=case(device);q,r,d=states(v)
    from warpSPHCore.crk.native.nt_volume import computeCRKVolume
    original=builtins.__import__
    def guarded(name,*a,**k):
        if name=='warp' or name.startswith('warp.'):raise AssertionError('native volume imported Warp')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    y=computeCRKVolume(q,p,d,None,r);bars=torch.autograd.grad(y,v,torch.ones_like(y),create_graph=True)
    assert not any(x.requires_grad for x in bars)
    torch.testing.assert_close(y,functions('native')(*v),rtol=5e-8,atol=5e-8)


def test_explicit_product_uses_supplied_volumes(device):
    from warpSPHCore.crk.native.nt_jvp import computeCRKVolumeGeometryJVP
    q,r,d,p,g,v,states,functions=case(device);q,r,d=states(tuple(x.detach() for x in v))
    qt=sph.ParticleTangentState(q.positions.cos()*.03,q.supports*.02,None,None)
    supplied=q.supports.new_tensor([.21,.67])
    actual=computeCRKVolumeGeometryJVP(q,d,p.kernel,g,supplied,qt,r,operationMode=p.operationMode)
    unit=list(x.detach() for x in v);unit[4]=torch.ones_like(unit[4]);directions=tuple(torch.zeros_like(x) for x in unit)
    directions=list(directions);directions[0]=qt.positions;directions[2]=qt.supports
    from _support.density_contract import oracle as sum_equation
    _,ds=torch.autograd.functional.jvp(lambda *x:sum_equation(x,d,p,g),tuple(unit),tuple(directions))
    torch.testing.assert_close(actual,-ds*supplied*supplied,rtol=5e-8,atol=5e-8)
    product=computeCRKVolumeGeometryJVP(q,d,p.kernel,g,supplied.requires_grad_(),qt,r,operationMode=p.operationMode)
    torch.testing.assert_close(torch.autograd.grad(product,supplied,torch.ones_like(product))[0],-2*ds*supplied,rtol=5e-8,atol=5e-8)


@pytest.mark.gpu
def test_current_stream_capture_reverse_and_products(device):
    if device!='cuda':pytest.skip('CUDA execution contract')
    from warpSPHCore.crk.native.nt_jvp import computeCRKVolumeGeometryJVP
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32);run=functions('native');t=tuple((x.detach().cos()*.03).requires_grad_() for x in v)
        supplied=v[2].detach().clone().requires_grad_()
        def products():
            y=run(*v);bars=torch.autograd.grad(y,v,torch.ones_like(y))
            qi,rj,domain=states(v)
            explicit=computeCRKVolumeGeometryJVP(qi,domain,p.kernel,g,supplied,sph.ParticleTangentState(t[0],t[2],None,None),rj,sph.ParticleTangentState(t[1],t[3],None,None),operationMode=p.operationMode)
            explicit_bars=torch.autograd.grad(explicit,(*v,t[0],t[1],t[2],t[3],supplied),torch.ones_like(explicit),allow_unused=True)
            explicit_bars=tuple(torch.zeros_like(x) if z is None else z for x,z in zip((*v,t[0],t[1],t[2],t[3],supplied),explicit_bars))
            with torch.autograd.forward_ad.dual_level():
                product=torch.autograd.forward_ad.unpack_dual(run(*[torch.autograd.forward_ad.make_dual(x.detach(),z) for x,z in zip(v,t)])).tangent
            return y,*bars,explicit,*explicit_bars,product
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[0][:,1].add_(.01);v[2].mul_(1.03);supplied.mul_(1.07)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,wanted):torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-4)
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('support',list(sph.SupportScheme))
def test_owned_rows_primal_and_inherited_ad_guard(device,support):
    from test_row_chunks import sample
    from warpSPHCore.crk.native.nt_volume import computeCRKVolume
    particles,domain,policy=sample(device,count=9)
    properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,supportMode=support)
    chunks=sph.radiusSearchRowChunks(particles,domain,policy=replace(policy,maxRows=2),mode=support)
    flat=sph.radiusSearch(particles,domain,mode=support)
    torch.testing.assert_close(computeCRKVolume(particles,properties,domain,chunks),
                               computeCRKVolume(particles,properties,domain,flat),rtol=5e-8,atol=5e-8)
    with pytest.raises(NotImplementedError,match='AD'):
        computeCRKVolume(replace(particles,positions=particles.positions.clone().requires_grad_()),properties,domain,chunks)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_explicit_product_reverse_gradcheck(device,kernel):
    from warpSPHCore.crk.native.nt_jvp import computeCRKVolumeGeometryJVP
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel)
    v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
    directions=tuple((x.detach().cos()*.03).requires_grad_() for x in (v[0],v[1],v[2],v[3]))
    supplied=v[2].detach().clone().requires_grad_()
    def run(*x):
        qi,rj,domain=states(x[:10])
        return computeCRKVolumeGeometryJVP(qi,domain,p.kernel,g,x[14],sph.ParticleTangentState(x[10],x[12],None,None),
            rj,sph.ParticleTangentState(x[11],x[13],None,None),p.operationMode)
    assert torch.autograd.gradcheck(run,(*v,*directions,supplied),fast_mode=True,atol=5e-5,rtol=8e-4)


@pytest.mark.parametrize('role',[4,7,8,9])
def test_explicit_sole_unused_state_and_frozen_bounds(device,role):
    from warpSPHCore.crk.native.nt_jvp import computeCRKVolumeGeometryJVP
    q,r,d,p,g,v,states,functions=case(device)
    v=[x.detach() for x in v];v[role].requires_grad_();q,r,d=states(v)
    qt=sph.ParticleTangentState(torch.ones_like(q.positions)*.03,None,None,None)
    V=torch.ones_like(q.supports)
    y=computeCRKVolumeGeometryJVP(q,d,p.kernel,g,V,qt,r,operationMode=p.operationMode)
    torch.testing.assert_close(torch.autograd.grad(y.sum(),v[role])[0],torch.zeros_like(v[role]))
    with torch.autograd.forward_ad.dual_level():
        domain=replace(d,min=torch.autograd.forward_ad.make_dual(d.min,torch.ones_like(d.min)))
        z=computeCRKVolumeGeometryJVP(q,domain,p.kernel,g,V,qt,r,operationMode=p.operationMode)
        assert torch.autograd.forward_ad.unpack_dual(z).tangent is None


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_explicit_sparse_strided_deterministic_reverse(device,index,offset):
    from warpSPHCore.crk.native.nt_jvp import computeCRKVolumeGeometryJVP
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset))
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32);g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_()
    v=tuple(strided(x) if i not in (5,6) else x.detach() for i,x in enumerate(v))
    directions=tuple(strided(x.detach().cos()*.03) for x in (v[0],v[1],v[2],v[3]));V=strided(v[2].detach())
    def run(*x):
        qi,rj,domain=states(x[:10])
        return computeCRKVolumeGeometryJVP(qi,domain,p.kernel,g,x[14],sph.ParticleTangentState(x[10],x[12],None,None),
            rj,sph.ParticleTangentState(x[11],x[13],None,None),p.operationMode)
    values=(*v,*directions,V);prior=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        assert torch.autograd.gradcheck(run,values,fast_mode=True,atol=5e-5,rtol=8e-4)
        previous=None
        for _ in range(3):
            out=run(*values);bars=torch.autograd.grad(out,[x for x in values if x.requires_grad],torch.ones_like(out))
            if previous is not None:
                for a,b in zip(bars,previous):torch.testing.assert_close(a,b,rtol=0,atol=0)
            previous=bars
    finally:torch.use_deterministic_algorithms(prior)
    out=run(*values);g.paddedJ[2]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):torch.autograd.grad(out,V,torch.ones_like(out))


def test_ordinary_reverse_materializes_efficient_zero_cotangent(device):
    *_,values,states,functions=case(device)
    y=functions('native')(*values)
    seed=torch._efficientzerotensor(y.shape,device=y.device,dtype=y.dtype)
    bars=torch.autograd.grad(y,[x for x in values if x.requires_grad],seed,allow_unused=True)
    for bar in bars:
        if bar is not None:torch.testing.assert_close(bar,torch.zeros_like(bar),rtol=0,atol=0)
