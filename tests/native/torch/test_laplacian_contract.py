"""Laplacian's four estimators, clipping and complete first-order products."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.laplacian_contract import case,oracle
from test_curl_contract import check_products


@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_variant_reverse_and_forward_gradcheck(device,kernel,mode,corrected):
    index=list(sph.KernelFunctions).index(kernel);dim=1+index%3
    field_shape=(dim,2,3) if mode==sph.LaplacianScheme.Dot else (1+index%5,)
    q,r,d,p,g,v,states,functions=case(device,dim,kernel=kernel,support=list(sph.SupportScheme)[index%6],
        corrected=corrected,field_shape=field_shape,scheme=list(sph.GradientScheme)[index%4],mode=mode)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=True,atol=6e-5,rtol=8e-4,check_undefined_grad=True)


@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('positive',[False,True])
@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
def test_joint_products_match_independent_equation(device,scheme,mode,positive,corrected):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(2,3,2),corrected=corrected,scheme=scheme,mode=mode,positive=positive)
    check_products(v,functions('native'),lambda *x:oracle(x,d,p,g,corrected),4e-8)


@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
def test_positive_divergence_reverse_forward_gradcheck(device,mode,corrected):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(2,5),corrected=corrected,mode=mode,positive=True)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=True,atol=6e-5,rtol=8e-4)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
def test_independent_query_reference_field_axes(device,mode):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(2,3),corrected=True,mode=mode)
    v=list(v);v[10]=v[10].detach().reshape(3,3,2).requires_grad_();v=tuple(v)
    assert functions('native')(*v).shape==(2,2,3)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=True,atol=6e-5,rtol=8e-4)


@pytest.mark.parametrize('operation',[sph.WarpOperation.Gradient,sph.WarpOperation.Divergence,sph.WarpOperation.Curl,sph.WarpOperation.Laplacian])
@pytest.mark.parametrize('role',['fields','volume','crk'])
def test_fixed_geometry_products_do_not_read_singular_hessian(device,operation,role):
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,field_shape=(2,),corrected=True,kernel=sph.KernelFunctions.ViscosityKernel)
    v=[x.detach() for x in v];v[0]=torch.full_like(v[0],1e-15);v[1]=torch.zeros_like(v[1])
    p=replace(p,operation=operation);run=functions('native',p);q,r,d=states(v[:10])
    options=dict(referenceParticles=r,adjacency=g,queryValues=v[14],referenceValues=v[10],referenceVolumes=v[13],
        crkState=sph.CRKState(v[11],v[12],v[15],v[16]),gradHState=sph.GradHState(v[19],v[17]),renormalizationState=v[18])
    directions=[None]*len(v)
    if role=='fields':
        directions[14]=torch.arange(v[14].numel(),device=device,dtype=v[14].dtype).reshape_as(v[14])*.01+.03
        directions[10]=torch.arange(v[10].numel(),device=device,dtype=v[10].dtype).reshape_as(v[10])*.02+.07
        options.update(tangentQueryValues=directions[14],tangentReferenceValues=directions[10])
        values=list(v);values[14]=directions[14];values[10]=directions[10]
        expected=run(*values)
    elif role=='volume':
        directions[13]=v[13]*.03;options['tangentReferenceVolumes']=directions[13];expected=run(*v)*.03
    else:
        directions[11]=v[11]*.03;options['crkTangentState']=sph.CRKTangentState(directions[11],None,None,None)
        # The CRK gradient has a gradA term independent of A.
        values=list(v);values[15]=torch.zeros_like(v[15]);expected=run(*values)*.03
    assert torch.isfinite(expected).all()
    actual=warpOperationJVP(q,p,d,**options)
    torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-3)
    with torch.autograd.forward_ad.dual_level():
        dual=[torch.autograd.forward_ad.make_dual(x,t) if t is not None else x for x,t in zip(v,directions)]
        product=torch.autograd.forward_ad.unpack_dual(run(*dual)).tangent
    if product is None:product=torch.zeros_like(expected)
    torch.testing.assert_close(product,expected,rtol=2e-3,atol=2e-3)


@pytest.mark.parametrize('slot',[7,11,12,15,16,18])
@pytest.mark.parametrize('singular',[False,True])
def test_naive_ignored_roles_have_zero_reverse_and_dual_products(device,slot,singular):
    q,r,d,p,g,v,states,functions=case(device,corrected=True,mode=sph.LaplacianScheme.Naive,
        kernel=sph.KernelFunctions.ViscosityKernel if singular else sph.KernelFunctions.Wendland2)
    values=[x.detach().requires_grad_(i==slot) for i,x in enumerate(v)]
    if singular:values[0]=torch.zeros_like(values[0]);values[1]=torch.zeros_like(values[1])
    run=functions('native');y=run(*values)
    torch.testing.assert_close(torch.autograd.grad(y,values[slot],torch.ones_like(y))[0],torch.zeros_like(values[slot]))
    with torch.autograd.forward_ad.dual_level():
        dual=list(values);dual[slot]=torch.autograd.forward_ad.make_dual(values[slot],torch.ones_like(values[slot]))
        product=torch.autograd.forward_ad.unpack_dual(run(*dual)).tangent
    torch.testing.assert_close(product,torch.zeros_like(y))


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('positive',[False,True])
def test_explicit_joint_corrected_jvp(device,mode,positive):
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,field_shape=(2,3,2),corrected=True,mode=mode,positive=positive)
    q,r,d=states(v[:10]);t=list(x.detach().cos()*.03 for x in v)
    t[17]=torch.zeros_like(t[17]);t[19]=torch.zeros_like(t[19])
    options=dict(referenceParticles=r,queryValues=v[14],referenceValues=v[10],adjacency=g,
        tangentQueryValues=t[14],tangentReferenceValues=t[10],referenceVolumes=v[13],tangentReferenceVolumes=t[13],
        queryTangentState=sph.ParticleTangentState(t[0],t[2],t[7],t[8]),referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],t[9]),
        crkState=sph.CRKState(v[11],v[12],v[15],v[16]),crkTangentState=sph.CRKTangentState(t[11],t[12],t[15],t[16]),
        gradHState=sph.GradHState(v[19],v[17]),renormalizationState=v[18],renormalizationTangentState=t[18])
    got=warpOperationJVP(q,p,d,**options)
    _,want=torch.autograd.functional.jvp(lambda *x:oracle(x,d,p,g,True),v,tuple(t))
    torch.testing.assert_close(got,want,rtol=4e-8,atol=4e-8)


import gc

@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_prefix_duplicates_empty_row_noncontiguous_and_deterministic(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,graph_types=(index,offset),field_shape=(5,2),corrected=True)
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32)
    g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_()
    v=tuple(strided(x) for x in v);seed=v[10].new_tensor([[.3,-.2,.1,-.7,.9],[-.5,.4,.1,.2,.8]])[:,:,None].repeat(1,1,2)
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
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,2));q,r,d=states(v[:10]);nq,nr=sizes
    q=replace(q,**{n:getattr(q,n)[:nq] for n in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{n:getattr(r,n)[:nr] for n in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(nq,device=device,dtype=torch.int32),torch.zeros(nq,device=device,dtype=torch.int64),nq,nr)
    output=sphOperation(q,p,d,referenceParticles=r,queryValues=v[14][:nq],referenceValues=v[10][:nr],adjacency=g,backend='native')
    assert output.shape==(nq,5,2)
    for gradient in torch.autograd.grad(output,v,torch.ones_like(output),allow_unused=True):
        if gradient is not None:assert not gradient.count_nonzero()
    unused=q.masses.detach().clone().requires_grad_()
    q=replace(q,positions=q.positions.detach(),supports=q.supports.detach(),densities=q.densities.detach(),masses=unused)
    r=replace(r,positions=r.positions.detach(),supports=r.supports.detach(),densities=r.densities.detach(),masses=r.masses.detach())
    output=sphOperation(q,p,replace(d,min=d.min.detach(),max=d.max.detach()),referenceParticles=r,queryValues=v[14][:nq].detach(),referenceValues=v[10][:nr].detach(),adjacency=g,backend='native')
    torch.testing.assert_close(torch.autograd.grad(output,unused,torch.ones_like(output))[0],torch.zeros_like(unused))


def test_native_search_and_reference_first_order_boundary(device,monkeypatch):
    import builtins
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,2))
    original=builtins.__import__
    def guarded(name,*a,**k):
        if 'wp_laplacian' in name:pytest.fail('native laplacian imported Warp')
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
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,field_shape=(5,2),corrected=True)
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


def test_three_step_laplacian_feedback_products(device):
    q,r,d,p,g,v,states,functions=case(device,field_shape=(5,2),corrected=True)
    def rollout(evaluate,*values):
        position=values[0]
        for _ in range(3):
            field=evaluate(position,*values[1:])
            position=position+.002*field.reshape(len(position),-1).sum(-1)[:,None].tanh()*position.cos()
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
