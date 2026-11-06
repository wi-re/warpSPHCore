"""Density position-block products, frozen geometry and higher Torch scope."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.density_block_contract import case,oracle,smooth_matrices
from test_curl_contract import check_products


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('omit_reference',[False,True])
def test_all_kernel_reverse_forward_and_gradgradcheck(device,kernel,omit_reference):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,functions=case(device,1+index%3,kernel=kernel,
        support=list(sph.SupportScheme)[index%6],omit_reference=omit_reference)
    fn=functions('native')
    assert torch.autograd.gradcheck(fn,v,fast_mode=True,check_forward_ad=True,atol=5e-5,rtol=5e-4)
    assert torch.autograd.gradgradcheck(fn,v,fast_mode=True,atol=5e-5,rtol=5e-4)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('omit_reference',[False,True])
def test_joint_products_match_independent_equation(device,dimension,omit_reference):
    q,r,d,p,g,v,functions=case(device,dimension,omit_reference=omit_reference)
    H=smooth_matrices(q,r,d,p)
    check_products(v,functions('native'),lambda *x:oracle(x,q,r,d,p,g,H),4e-8)


def derivatives(fn,v):
    output=fn(*v);seed=output.detach().cos();directions=tuple(x.detach().cos()*.03 for x in v)
    first=torch.autograd.grad((output.sin()*seed).sum(),v,create_graph=True)
    second=torch.autograd.grad(sum((x*t).sum() for x,t in zip(first,directions)),v,create_graph=True)
    third=torch.autograd.grad(sum((x*t).sum() for x,t in zip(second,directions)),v)
    return first,second,third


@pytest.mark.parametrize('dimension',[1,2,3])
def test_hessians_third_derivatives_and_batched_vjps(device,dimension):
    q,r,d,p,g,v,functions=case(device,dimension);H=smooth_matrices(q,r,d,p)
    expected=lambda *x:oracle(x,q,r,d,p,g,H)
    actual=functions('native')
    for actual_level,expected_level in zip(derivatives(actual,v),derivatives(expected,v)):
        for a,b in zip(actual_level,expected_level):torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
    for fn in (actual,expected):
        output=fn(*v);seed=torch.stack((output.detach().cos(),output.detach().sin(),torch.ones_like(output)))
        result=torch.autograd.grad(output,v,seed,is_grads_batched=True,create_graph=True)
        if fn is actual:got=result
        else:
            for a,b in zip(got,result):torch.testing.assert_close(a,b,rtol=3e-8,atol=3e-8)
    for fn in (actual,expected):
        jacobian=torch.autograd.functional.jacobian(fn,v,vectorize=True,strategy='forward-mode')
        if fn is actual:forward_jacobian=jacobian
        else:
            for a,b in zip(forward_jacobian,jacobian):torch.testing.assert_close(a,b,rtol=4e-8,atol=4e-8)
    for vectorize in (False,True):
        want=torch.autograd.functional.hessian(lambda *x:expected(*x).square().sum(),v,vectorize=vectorize)
        got=torch.autograd.functional.hessian(lambda *x:actual(*x).square().sum(),v,vectorize=vectorize)
        for a,b in zip(got,want):
            for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=3e-7,atol=3e-7)


@pytest.mark.parametrize('dimension',[1,2,3])
def test_forward_reverse_compositions(device,dimension):
    q,r,d,p,g,v,functions=case(device,dimension);H=smooth_matrices(q,r,d,p)
    expected=lambda *x:oracle(x,q,r,d,p,g,H)
    directions=tuple(x.detach().cos()*.03 for x in v)
    for forward_over_reverse in (False,True):
        for fn in (functions('native'),expected):
            with torch.autograd.forward_ad.dual_level():
                dual=[torch.autograd.forward_ad.make_dual(x,t) for x,t in zip(v,directions)]
                output=fn(*dual)
                if forward_over_reverse:
                    grads=torch.autograd.grad(output.sum(),dual)
                    result=[torch.autograd.forward_ad.unpack_dual(x).tangent for x in grads]
                else:
                    tangent=torch.autograd.forward_ad.unpack_dual(output).tangent
                    result=torch.autograd.grad(tangent.sum(),v)
            if fn is expected:
                for a,b in zip(got,result):torch.testing.assert_close(a,b,rtol=3e-8,atol=3e-8)
            else:got=result


@pytest.mark.parametrize('role',['query_positions','reference_positions','query_supports','reference_supports','minimum','maximum'])
def test_geometry_autograd_metadata_is_frozen(device,role):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP
    q,r,d,p,g,v,functions=case(device);v=tuple(x.detach() for x in v)
    if role.startswith('query_'):
        name=role.removeprefix('query_');q=replace(q,**{name:getattr(q,name).clone().requires_grad_()})
    elif role.startswith('reference_'):
        name=role.removeprefix('reference_');r=replace(r,**{name:getattr(r,name).clone().requires_grad_()})
    else:
        name='min' if role=='minimum' else 'max';d=replace(d,**{name:getattr(d,name).clone().requires_grad_()})
    result=computeSPHDensityPositionHVP(q,d,p.kernel,p.supportMode,g,v[1],replace(r,masses=v[0]),v[2],p.operationMode)
    assert not result.requires_grad
    torch.testing.assert_close(result,functions('native')(*v),rtol=0,atol=0)


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_empty_row_and_deterministic_products(device,index,offset):
    q,r,d,p,g,v,functions=case(device,graph_types=(index,offset))
    g.paddedJ=torch.tensor([-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32)
    g.edgeOffsets=torch.tensor([1,5],device=device,dtype=offset)
    def csr_view(x):
        storage=torch.empty((x.numel(),2),device=x.device,dtype=x.dtype);storage[:,0]=x
        return storage[:,0]
    g.paddedJ=csr_view(g.paddedJ);g.numNeighbors=csr_view(g.numNeighbors);g.edgeOffsets=csr_view(g.edgeOffsets)
    def strided(x):
        storage=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);storage[...,0]=x.detach()
        return storage[...,0].requires_grad_()
    v=tuple(strided(x) for x in v);H=smooth_matrices(q,r,d,p)
    expected=oracle(v,q,r,d,p,g,H);seed=expected.detach().cos()
    wanted=torch.autograd.grad(expected,v,seed)
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for _ in range(3):
            y=functions('native')(*v);got=torch.autograd.grad(y,v,seed)
            torch.testing.assert_close(y,oracle(v,q,r,d,p,g,H),rtol=4e-8,atol=4e-8)
            for a,b in zip(got,wanted):torch.testing.assert_close(a,b,rtol=4e-8,atol=4e-8)
            if _==0:first=got
            else:
                for a,b in zip(first,got):torch.testing.assert_close(a,b,rtol=0,atol=0)
        y=functions('native')(*v)
        batched=torch.autograd.grad(y,v,torch.stack((seed,seed*.3)),is_grads_batched=True,create_graph=True)
        for a,b in zip(batched,wanted):torch.testing.assert_close(a,torch.stack((b,b*.3)),rtol=4e-8,atol=4e-8)
    finally:torch.use_deterministic_algorithms(previous)
    y=functions('native')(*v);g.paddedJ[1]=1
    with pytest.raises(RuntimeError,match='CSR topology changed'):
        torch.autograd.grad(y,v,torch.ones_like(y))


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_populations_and_products(device,sizes):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP
    q,r,d,p,g,v,functions=case(device);nq,nr=sizes
    q=replace(q,**{name:getattr(q,name)[:nq] for name in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{name:getattr(r,name)[:nr] for name in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(nq,device=device,dtype=torch.int32),torch.zeros(nq,device=device,dtype=torch.int64),nq,nr)
    values=tuple(x.detach().requires_grad_() for x in (r.masses,v[1][:nq],v[2][:nr]))
    def run(m,a,b):return computeSPHDensityPositionHVP(q,d,p.kernel,p.supportMode,g,a,replace(r,masses=m),b,p.operationMode)
    for graph in (False,True):
        y=run(*values);grads=torch.autograd.grad(y.sum(),values,create_graph=graph)
        assert y.shape==(nq,2) and not y.count_nonzero()
        for value in grads:assert not value.count_nonzero()
    with torch.autograd.forward_ad.dual_level():
        y=run(*(torch.autograd.forward_ad.make_dual(x.detach(),torch.ones_like(x)) for x in values))
        product=torch.autograd.forward_ad.unpack_dual(y).tangent
    assert product.shape==(nq,2) and not product.count_nonzero()


def test_shared_population_aliases_and_translation_nullspace(device):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP
    q,r,d,p,g,v,functions=case(device);n=len(q.positions)
    g=sph.AdjacencyList(None,torch.arange(n,device=device).repeat(n),torch.full((n,),n,device=device,dtype=torch.int32),torch.arange(n,device=device)*n,n,n)
    mass=q.masses.clone().requires_grad_();vector=v[1].detach().clone().requires_grad_()
    def run(m,x):return computeSPHDensityPositionHVP(replace(q,masses=m),d,p.kernel,p.supportMode,g,x,
        tangentReferencePositions=x,operationMode=p.operationMode)
    assert torch.autograd.gradcheck(run,(mass,vector),fast_mode=True,check_forward_ad=True)
    assert torch.autograd.gradgradcheck(run,(mass,vector),fast_mode=True)
    translated=run(mass,torch.ones_like(vector))
    torch.testing.assert_close(translated,torch.zeros_like(translated),rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(translated.sum(),mass)[0],torch.zeros_like(mass),rtol=0,atol=0)


def test_native_search_no_warp_and_functional_scope(device,monkeypatch):
    import builtins
    q,r,d,p,g,v,functions=case(device)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if 'wp_densityHVP' in name:pytest.fail('native position block imported Warp')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    torch.testing.assert_close(functions('native',adjacency=None)(*v),functions('native')(*v),rtol=4e-8,atol=4e-8)
    assert torch.autograd.gradgradcheck(functions('native'),v,fast_mode=True)
    with pytest.raises(RuntimeError,match='setup_context'):
        torch.func.vjp(lambda mass:functions('native')(mass,*v[1:]),v[0])


@pytest.mark.gpu
def test_current_stream_capture_first_higher_batched_and_dual_products(device):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,functions=case(device,dtype=torch.float32);run=functions('native')
        seed=v[1].new_full(v[1].shape,.2);batch=torch.stack((seed,seed*.3));directions=tuple(x.detach().cos()*.03 for x in v)
        def products():
            y=run(*v);bars=torch.autograd.grad(y,v,seed)
            higher=torch.autograd.grad(run(*v).square().sum(),v,create_graph=True)
            second=torch.autograd.grad(sum((x*t).sum() for x,t in zip(higher,directions)),v)
            batched=torch.autograd.grad(run(*v),v,batch,is_grads_batched=True)
            with torch.autograd.forward_ad.dual_level():
                dual=[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,directions)]
                product=torch.autograd.forward_ad.unpack_dual(run(*dual)).tangent
            return (y,*bars,*second,*batched,product)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[0].mul_(1.04);v[1].add_(.01);q.positions[:,1].add_(.002)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):expected=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip(captured,expected):torch.testing.assert_close(a,b,rtol=3e-3,atol=3e-3)
    finally:
        if enabled:gc.enable()


def test_row_chunks_preserve_global_self_ids_and_frozen_metadata(device):
    from test_row_chunks import sample
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP
    particles,domain,policy=sample(device)
    support=sph.SupportScheme.SuperSymmetric;kernel=sph.KernelFunctions.Wendland2
    chunks=sph.radiusSearchRowChunks(particles,domain,policy=policy,mode=support)
    flat=sph.radiusSearch(particles,domain,mode=support)
    assert len(chunks.chunks)>1
    vector=particles.positions.cos()*.03
    for right in (None,vector):
        expected=computeSPHDensityPositionHVP(particles,domain,kernel,support,flat,vector,tangentReferencePositions=right)
        actual=computeSPHDensityPositionHVP(particles,domain,kernel,support,chunks,vector,tangentReferencePositions=right)
        torch.testing.assert_close(actual,expected,rtol=4e-8,atol=4e-8)
    query=replace(particles,positions=particles.positions.clone().requires_grad_(),
        supports=particles.supports.clone().requires_grad_(),masses=particles.masses.clone().requires_grad_(),
        densities=particles.densities.clone().requires_grad_())
    reference=replace(particles,positions=particles.positions.clone().requires_grad_(),
        supports=particles.supports.clone().requires_grad_(),densities=particles.densities.clone().requires_grad_())
    actual=computeSPHDensityPositionHVP(query,domain,kernel,support,chunks,vector,reference,vector)
    torch.testing.assert_close(actual,expected,rtol=4e-8,atol=4e-8)
    assert not actual.requires_grad
    with pytest.raises(NotImplementedError,match='Row-chunk'):
        computeSPHDensityPositionHVP(query,domain,kernel,support,chunks,vector.requires_grad_(),reference,vector)


def test_self_population_omitted_reference_vector_is_zero(device):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP
    q,r,d,p,g,v,functions=case(device);n=len(q.positions)
    g=sph.AdjacencyList(None,torch.arange(n,device=device).repeat(n),torch.full((n,),n,device=device,dtype=torch.int32),torch.arange(n,device=device)*n,n,n)
    mass=q.masses.clone().requires_grad_();vector=torch.ones_like(q.positions).requires_grad_()
    H=smooth_matrices(q,q,d,p)
    def run(m,x):return computeSPHDensityPositionHVP(replace(q,masses=m),d,p.kernel,p.supportMode,g,x,operationMode=p.operationMode)
    check_products((mass,vector),run,lambda m,x:oracle((m,x),q,q,d,p,g,H),4e-8)
    assert run(mass,vector).count_nonzero()


@pytest.mark.parametrize('role',['query_positions','reference_positions','query_supports','reference_supports','minimum','maximum'])
def test_geometry_forward_duals_are_frozen(device,role):
    from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP
    q,r,d,p,g,v,functions=case(device);v=tuple(x.detach() for x in v)
    with torch.autograd.forward_ad.dual_level():
        if role.startswith('query_'):
            name=role.removeprefix('query_');value=getattr(q,name)
            q=replace(q,**{name:torch.autograd.forward_ad.make_dual(value,torch.ones_like(value)*.03)})
        elif role.startswith('reference_'):
            name=role.removeprefix('reference_');value=getattr(r,name)
            r=replace(r,**{name:torch.autograd.forward_ad.make_dual(value,torch.ones_like(value)*.03)})
        else:
            name='min' if role=='minimum' else 'max';value=getattr(d,name)
            d=replace(d,**{name:torch.autograd.forward_ad.make_dual(value,torch.ones_like(value)*.03)})
        actual=computeSPHDensityPositionHVP(q,d,p.kernel,p.supportMode,g,v[1],replace(r,masses=v[0]),v[2],p.operationMode)
        primal,tangent=torch.autograd.forward_ad.unpack_dual(actual)
    assert tangent is None
    torch.testing.assert_close(primal,functions('native')(*v),rtol=0,atol=0)
