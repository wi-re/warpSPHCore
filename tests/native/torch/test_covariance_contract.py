"""Covariance's geometry/correction products and discrete-count contract."""
from dataclasses import replace
import pytest
import torch
import warpSPHCore as sph
from _support.covariance_contract import case,oracle
from test_curl_contract import check_products


@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_all_kernel_reverse_forward_gradcheck(device,kernel,corrected):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,kernel=kernel,support=list(sph.SupportScheme)[index%6],corrected=corrected)
    assert torch.autograd.gradcheck(functions('native'),v,fast_mode=True,check_forward_ad=True,atol=5e-5,rtol=8e-4,check_undefined_grad=True)


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('counts',[False,True])
def test_joint_products_and_discrete_counts(device,dimension,corrected,counts):
    q,r,d,p,g,v,states,functions=case(device,dimension,corrected=corrected,counts=counts)
    actual=functions('native')
    if counts:
        C,N=actual(*v);want,number=oracle(v,d,p,g,corrected,True)
        assert N.dtype==torch.int32 and not N.requires_grad
        torch.testing.assert_close(N,number,rtol=0,atol=0)
        with torch.autograd.forward_ad.dual_level():
            dual=[torch.autograd.forward_ad.make_dual(x.detach(),torch.cos(x.detach())*.03) for x in v]
            y,n=actual(*dual)
            assert torch.autograd.forward_ad.unpack_dual(n).tangent is None
        fn=lambda *x:actual(*x)[0]
    else:fn=actual
    check_products(v,fn,lambda *x:oracle(x,d,p,g,corrected),5e-8)


def test_counts_use_corrected_value_and_ignore_gradient_matrix(device):
    q,r,d,p,g,v,states,functions=case(device,corrected=True,counts=True)
    v=list(v);v[11]=v[11].new_tensor([-.9,0.]).requires_grad_();v[18]=-v[18];v=tuple(v)
    matrix,n=functions('native')(*v);expected,number=oracle(v,d,p,g,True,True)
    torch.testing.assert_close(matrix,expected,rtol=5e-8,atol=5e-8);torch.testing.assert_close(n,number)
    assert not n.count_nonzero()
    assert matrix[1].count_nonzero() # gradA survives even though A=0.


@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
def test_unread_fields_density_and_scheme_flags(device,scheme):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device);q,r,d=states(v[:10])
    expected=functions('native')(*v)
    p=replace(p,gradientMode=scheme,positiveDivergence=True,divergenceDotMode=True)
    actual=sphOperation(replace(q,densities=None),p,d,queryValues=v[0],referenceValues=v[1],
        referenceParticles=r,adjacency=g,gradHState=sph.GradHState(v[19],v[17]),backend='native')
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('role',['volume','A','matrix'])
def test_fixed_geometry_singular_coefficient_jvp(device,role):
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,corrected=True,kernel=sph.KernelFunctions.ViscosityKernel)
    v=[x.detach() for x in v];v[0]=torch.full_like(v[0],1e-16);v[1]=torch.zeros_like(v[1]);q,r,d=states(v[:10])
    options=dict(referenceParticles=r,adjacency=g,referenceVolumes=v[13],crkState=sph.CRKState(v[11],v[12],v[15],v[16]),renormalizationState=v[18])
    directions=[None]*len(v)
    if role=='volume':directions[13]=v[13]*.03;options['tangentReferenceVolumes']=directions[13]
    elif role=='A':directions[11]=v[11]*.03;options['crkTangentState']=sph.CRKTangentState(directions[11],None,None,None)
    else:directions[18]=v[18]*.03;options['renormalizationTangentState']=directions[18]
    expected_values=list(v)
    if role=='A':expected_values[15]=torch.zeros_like(v[15])
    expected=functions('native')(*expected_values)*.03
    assert torch.isfinite(expected).all()
    actual=warpOperationJVP(q,p,d,**options)
    torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-3)
    with torch.autograd.forward_ad.dual_level():
        dual=[torch.autograd.forward_ad.make_dual(x,t) if t is not None else x for x,t in zip(v,directions)]
        product=torch.autograd.forward_ad.unpack_dual(functions('native')(*dual)).tangent
    torch.testing.assert_close(product,expected,rtol=2e-3,atol=2e-3)

@pytest.mark.parametrize('index,offset',[(torch.int32,torch.int32),(torch.int32,torch.int64),(torch.int64,torch.int32),(torch.int64,torch.int64)])
def test_sparse_strided_empty_row_and_deterministic_products(device,index,offset):
    q,r,d,p,g,v,states,functions=case(device,corrected=True,graph_types=(index,offset),counts=True)
    g.paddedJ=torch.tensor([-1,-1,0,2,2,-1,-1],device=device,dtype=index)
    g.numNeighbors=torch.tensor([3,0],device=device,dtype=torch.int32);g.edgeOffsets=torch.tensor([2,6],device=device,dtype=offset)
    def strided(x):
        store=torch.empty((*x.shape,2),device=x.device,dtype=x.dtype);store[...,0]=x.detach()
        return store[...,0].requires_grad_()
    v=tuple(strided(x) for x in v);seed=v[0].new_tensor([.3,-.2,.1,-.7,.9,-.5,.4,.2]).reshape(2,2,2)
    expected,counts=oracle(v,d,p,g,True,True);wanted=torch.autograd.grad(expected,v,seed,allow_unused=True)
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for _ in range(3):
            actual,n=functions('native')(*v);got=torch.autograd.grad(actual,v,seed,allow_unused=True)
            torch.testing.assert_close(actual,expected,rtol=5e-8,atol=5e-8);torch.testing.assert_close(n,counts)
            for a,b,x in zip(got,wanted,v):torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=5e-8,atol=5e-8)
            if _==0:first=got
            else:
                for a,b in zip(first,got):
                    if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(previous)
    out,_=functions('native')(*v);g.paddedJ[2]=1
    with pytest.raises(RuntimeError,match='topology changed|modified by an inplace'):torch.autograd.grad(out,v,torch.ones_like(out),allow_unused=True)


@pytest.mark.parametrize('sizes',[(0,3),(2,0),(0,0)])
def test_empty_populations_and_sole_unused_mass(device,sizes):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device);q,r,d=states(v[:10]);nq,nr=sizes
    q=replace(q,**{n:getattr(q,n)[:nq].detach() for n in ('positions','supports','masses','kinds','densities')})
    r=replace(r,**{n:getattr(r,n)[:nr].detach() for n in ('positions','supports','masses','kinds','densities')})
    g=sph.AdjacencyList(None,torch.empty(0,device=device,dtype=torch.int64),torch.zeros(nq,device=device,dtype=torch.int32),torch.zeros(nq,device=device,dtype=torch.int64),nq,nr)
    unused=q.masses.clone().requires_grad_();q=replace(q,masses=unused)
    out,n=sphOperation(q,p,replace(d,min=d.min.detach(),max=d.max.detach()),referenceParticles=r,adjacency=g,covarianceReturnNumNeighbors=True,backend='native')
    assert out.shape==(nq,2,2) and n.shape==(nq,) and not n.requires_grad
    assert not out.count_nonzero() and not n.count_nonzero()
    torch.testing.assert_close(torch.autograd.grad(out,unused,torch.ones_like(out))[0],torch.zeros_like(unused))
    with torch.autograd.forward_ad.dual_level():
        q=replace(q,masses=torch.autograd.forward_ad.make_dual(unused.detach(),torch.ones_like(unused)))
        y,n=sphOperation(q,p,replace(d,min=d.min.detach(),max=d.max.detach()),referenceParticles=r,adjacency=g,covarianceReturnNumNeighbors=True,backend='native')
        torch.testing.assert_close(torch.autograd.forward_ad.unpack_dual(y).tangent,torch.zeros_like(out))


def test_explicit_joint_corrected_jvp(device):
    from warpSPHCore.native.api import warpOperationJVP
    q,r,d,p,g,v,states,functions=case(device,corrected=True);q,r,d=states(v[:10]);t=tuple(x.detach().cos()*.03 for x in v)
    options=dict(referenceParticles=r,adjacency=g,queryTangentState=sph.ParticleTangentState(t[0],t[2],t[7],t[8]),
        referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],t[9]),referenceVolumes=v[13],tangentReferenceVolumes=t[13],
        crkState=sph.CRKState(v[11],v[12],v[15],v[16]),crkTangentState=sph.CRKTangentState(t[11],t[12],t[15],t[16]),
        renormalizationState=v[18],renormalizationTangentState=t[18])
    actual=warpOperationJVP(q,p,d,**options)
    # Explicit products have no domain-bound tangent arguments.
    t=list(t);t[5]=torch.zeros_like(t[5]);t[6]=torch.zeros_like(t[6])
    _,expected=torch.autograd.functional.jvp(lambda *x:oracle(x,d,p,g,True),v,tuple(t))
    torch.testing.assert_close(actual,expected,rtol=5e-8,atol=5e-8)


def test_native_search_and_first_order_guards(device,monkeypatch):
    import builtins
    q,r,d,p,g,v,states,functions=case(device,counts=True)
    original=builtins.__import__
    def guarded(name,*a,**k):
        if 'wp_covariance' in name:pytest.fail('native covariance imported Warp')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    actual,n=functions('native',adjacency=None)(*v);expected,number=functions('native')(*v)
    torch.testing.assert_close(actual,expected);torch.testing.assert_close(n,number)
    first=torch.autograd.grad(actual,v[0],torch.ones_like(actual),create_graph=True)[0]
    assert not first.requires_grad
    with torch.autograd.forward_ad.dual_level():
        y,n=functions('native')(*[torch.autograd.forward_ad.make_dual(x,torch.ones_like(x)) for x in v])
        product=torch.autograd.forward_ad.unpack_dual(y).tangent
    with pytest.raises(NotImplementedError,match='first-order'):torch.autograd.grad(product.sum(),v[0])
    with pytest.raises(RuntimeError,match='setup_context'):torch.func.vjp(lambda x:functions('native')(x,*v[1:])[0],v[0])


def test_shared_population_mass_volume_aliases(device):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device);q=states(v[:10])[0]
    g=sph.AdjacencyList(None,torch.arange(2,device=device).repeat(2),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device),2,2)
    def run(x,h,m,rho):
        state=replace(q,positions=x,supports=h,masses=m,densities=rho)
        return sphOperation(state,p,d,queryVolumes=m,adjacency=g,backend='native')
    assert torch.autograd.gradcheck(run,(q.positions,q.supports,q.masses,q.densities),fast_mode=True,check_forward_ad=True)


def test_three_step_covariance_feedback_products(device):
    q,r,d,p,g,v,states,functions=case(device,corrected=True)
    def rollout(evaluate,*values):
        position=values[0]
        for _ in range(3):
            matrix=evaluate(position,*values[1:]);position=position+.002*matrix.sum((1,2))[:,None].tanh()*position.cos()
        return torch.cat((position.flatten(),matrix.flatten()))
    actual=lambda *x:rollout(functions('native'),*x)
    expected=lambda *x:rollout(lambda *z:oracle(z,d,p,g,True),*x)
    check_products(v,actual,expected,6e-8)


@pytest.mark.gpu
def test_current_stream_capture_counts_reverse_and_jvp(device):
    if device!='cuda':pytest.skip('CUDA execution contract')
    import gc
    gc.collect();enabled=gc.isenabled();gc.disable()
    try:
        q,r,d,p,g,v,states,functions=case(device,dtype=torch.float32,corrected=True,counts=True)
        run=functions('native');seed=v[0].new_full((2,2,2),.2);t=tuple(x.detach().cos()*.03 for x in v)
        def products():
            matrix,n=run(*v);bars=torch.autograd.grad(matrix,v,seed,allow_unused=True)
            with torch.autograd.forward_ad.dual_level():
                y,number=run(*[torch.autograd.forward_ad.make_dual(x.detach(),z) for x,z in zip(v,t)])
                product=torch.autograd.forward_ad.unpack_dual(y).tangent
            return matrix,n,bars,product
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):products()
        torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=products()
        with torch.no_grad():v[13].mul_(1.07);v[0][:,1].add_(.01);v[11].mul_(-1)
        graph.replay();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):wanted=products()
        torch.cuda.current_stream().wait_stream(stream)
        for a,b in zip((captured[0],captured[1],*captured[2],captured[3]),(wanted[0],wanted[1],*wanted[2],wanted[3])):
            if a is not None:torch.testing.assert_close(a,b,rtol=4e-4,atol=4e-4)
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('bundle',['state','tuple','tensor'])
def test_sole_unused_grad_h_zero_products(device,bundle):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device,dtype=torch.float64)
    q,r,d=states(tuple(x.detach() for x in v[:10]))
    omega=v[19].detach().clone().requires_grad_()
    def evaluate(value,backend):
        correction=sph.GradHState(value,value) if bundle=='state' else (value,value) if bundle=='tuple' else value
        return sphOperation(q,p,d,referenceParticles=r,adjacency=g,gradHState=correction,backend=backend)
    actual=evaluate(omega,'native')
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),omega)[0],torch.zeros_like(omega))
    with torch.autograd.forward_ad.dual_level():
        dual=torch.autograd.forward_ad.make_dual(omega.detach(),torch.ones_like(omega))
        product=torch.autograd.forward_ad.unpack_dual(evaluate(dual,'native')).tangent
    torch.testing.assert_close(product,torch.zeros_like(actual))
