"""Curl first-order products and inherited dimensional/packed layouts."""
from dataclasses import replace
import gc
import pytest
import torch
import warpSPHCore as sph
from _support.curl_contract import case,oracle


def check_products(v,actual,expected,tol):
    y=actual(*v);want=expected(*v);seed=y.detach().cos()
    torch.testing.assert_close(y,want,rtol=tol,atol=tol,equal_nan=True)
    got=torch.autograd.grad(y,v,seed,allow_unused=True)
    ref=torch.autograd.grad(want,v,seed,allow_unused=True)
    zero=lambda a,x:torch.zeros_like(x) if a is None else a
    for a,b,x in zip(got,ref,v):torch.testing.assert_close(zero(a,x),zero(b,x),rtol=tol,atol=tol,equal_nan=True)
    directions=tuple(x.detach().cos()*.03 for x in v)
    with torch.autograd.forward_ad.dual_level():
        product=torch.autograd.forward_ad.unpack_dual(actual(*[torch.autograd.forward_ad.make_dual(x.detach(),t) for x,t in zip(v,directions)])).tangent
    _,wanted=torch.autograd.functional.jvp(expected,v,directions)
    torch.testing.assert_close(product,wanted,rtol=tol,atol=tol,equal_nan=True)
    torch.testing.assert_close((product*seed).sum(),sum((a*t).sum() for a,t in zip(got,directions) if a is not None),rtol=tol,atol=tol)
    scalar=torch.autograd.grad(actual(*v).sum(),v,allow_unused=True)
    scalar_ref=torch.autograd.grad(expected(*v).sum(),v,allow_unused=True)
    for a,b,x in zip(scalar,scalar_ref,v):torch.testing.assert_close(zero(a,x),zero(b,x),rtol=tol,atol=tol)



@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_reverse_and_forward_gradcheck(device,kernel,corrected):
    index=list(sph.KernelFunctions).index(kernel);dim=1+index%3
    shape=(dim,2,5) if dim!=1 else (1,3)
    q,r,d,p,g,v,states,functions=case(device,dim,kernel=kernel,support=list(sph.SupportScheme)[index%6],
        corrected=corrected,field_shape=shape,scheme=list(sph.GradientScheme)[index%4])
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=True,atol=4e-5,rtol=5e-4,check_undefined_grad=True)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('corrected',[False,True])
def test_joint_products_match_independent_equation(device,dimension,scheme,corrected):
    q,r,d,p,g,v,states,functions=case(device,dimension,field_shape=(dimension,2,5),corrected=corrected,scheme=scheme)
    check_products(v,functions('native'),lambda *x:oracle(x,d,p,g,corrected),3e-8)


@pytest.mark.parametrize('dimension',[1,2,3])
def test_vector_scalar_matrix_and_packed_layouts(device,dimension):
    shapes=[(dimension,),(dimension,dimension),(dimension,2,3),(dimension,3,2)]
    if dimension==1:shapes.append(())
    elif dimension==2:shapes.extend(((5,),(3,2)))
    else:shapes.extend(((),(2,),(5,),(2,3,2)))
    for shape in shapes:
        q,r,d,p,g,v,states,functions=case(device,dimension,field_shape=shape,corrected=True)
        check_products(v,functions('native'),lambda *x:oracle(x,d,p,g,True),3e-8)


@pytest.mark.parametrize('dimension',[1,2,3])
def test_explicit_joint_corrected_jvp(device,dimension):
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,dimension,field_shape=(dimension,2,3),corrected=True)
    q,r,d=states(v[:10]);t=list(x.detach().cos()*.03 for x in v)
    t[17]=torch.zeros_like(t[17]);t[19]=torch.zeros_like(t[19])
    options=dict(referenceParticles=r,queryValues=v[14],referenceValues=v[10],adjacency=g,
        tangentQueryValues=t[14],tangentReferenceValues=t[10],referenceVolumes=v[13],tangentReferenceVolumes=t[13],
        queryTangentState=sph.ParticleTangentState(t[0],t[2],t[7],t[8]),referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],t[9]),
        crkState=sph.CRKState(v[11],v[12],v[15],v[16]),crkTangentState=sph.CRKTangentState(t[11],t[12],t[15],t[16]),
        gradHState=sph.GradHState(v[19],v[17]),renormalizationState=v[18],renormalizationTangentState=t[18])
    actual=warpOperationJVP(q,p,d,**options)
    _,want=torch.autograd.functional.jvp(lambda *x:oracle(x,d,p,g,True),v,tuple(t))
    torch.testing.assert_close(actual,want,rtol=4e-8,atol=4e-8)


@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_prefix_duplicates_empty_row_noncontiguous_and_deterministic(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset),field_shape=(2,5),corrected=True)
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32)
    g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_()
    v=tuple(strided(x) for x in v);seed=oracle(v,d,p,g,True).detach().cos().contiguous()
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
    q,r,d,p,g,v,states,functions=case(device,field_shape=(2,5));q,r,d=states(v[:10]);nq,nr=sizes
    q=replace(q,**{n:getattr(q,n)[:nq] for n in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{n:getattr(r,n)[:nr] for n in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(nq,device=device,dtype=torch.int32),torch.zeros(nq,device=device,dtype=torch.int64),nq,nr)
    output=sphOperation(q,p,d,referenceParticles=r,queryValues=v[14][:nq],referenceValues=v[10][:nr],adjacency=g,backend='native')
    assert output.shape==(nq,2)
    for gradient in torch.autograd.grad(output,v,torch.ones_like(output),allow_unused=True):
        if gradient is not None:assert not gradient.count_nonzero()
    unused=q.masses.detach().clone().requires_grad_()
    q=replace(q,positions=q.positions.detach(),supports=q.supports.detach(),densities=q.densities.detach(),masses=unused)
    r=replace(r,positions=r.positions.detach(),supports=r.supports.detach(),densities=r.densities.detach(),masses=r.masses.detach())
    output=sphOperation(q,p,replace(d,min=d.min.detach(),max=d.max.detach()),referenceParticles=r,queryValues=v[14][:nq].detach(),referenceValues=v[10][:nr].detach(),adjacency=g,backend='native')
    torch.testing.assert_close(torch.autograd.grad(output,unused,torch.ones_like(output))[0],torch.zeros_like(unused))




def test_native_search_and_reference_first_order_boundary(device,monkeypatch):
    import builtins
    q,r,d,p,g,v,states,functions=case(device,field_shape=(2,5))
    original=builtins.__import__
    def guarded(name,*a,**k):
        if 'wp_curl' in name:pytest.fail('native curl imported Warp')
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
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,field_shape=(2,5),corrected=True)
        run=functions('native');seed=v[10].new_full((2,2),.2);directions=tuple(torch.cos(x.detach())*.03 for x in v)
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


def test_three_step_curl_feedback_products(device):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(2,5),corrected=True)
    def rollout(evaluate,*values):
        position=values[0]
        for _ in range(3):
            field=evaluate(position,*values[1:])
            position=position+.002*field.sum(-1)[:,None].tanh()*position.cos()
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




def test_shared_population_aliases_and_ignored_roles(device):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device);q=states(v[:10])[0]
    g=sph.AdjacencyList(None,torch.arange(2,device=device).repeat(2),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device),2,2)
    def run(x,h,m,rho,field):
        state=replace(q,positions=x,supports=h,masses=m,densities=rho)
        return sphOperation(state,p,d,queryValues=field,queryVolumes=m,adjacency=g,backend='native')
    field=v[14];values=(q.positions,q.supports,q.masses,q.densities,field)
    assert torch.autograd.gradcheck(run,values,fast_mode=True,check_forward_ad=True)


@pytest.mark.parametrize('slot',[17,19])
def test_sole_omega_dependency(device,slot):
    q,r,d,p,g,v,states,functions=case(device,corrected=True)
    values=tuple(x.detach().requires_grad_(i==slot) for i,x in enumerate(v))
    y=functions('native')(*values);wanted=oracle(values,d,p,g,True)
    torch.testing.assert_close(torch.autograd.grad(y,values[slot],torch.ones_like(y))[0],
        torch.autograd.grad(wanted,values[slot],torch.ones_like(wanted))[0],rtol=3e-8,atol=3e-8)
