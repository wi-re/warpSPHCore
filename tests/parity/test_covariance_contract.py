"""Covariance matrix/counts and first-order products versus Warp."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.coreOperations.native.wrapper import sphOperation
from warpSPHCore.native.api import warpOperationJVP as native_jvp
from _support.covariance_contract import case


@pytest.fixture(scope='module',autouse=True)
def initialize_reference():wp.init()


def compare(values,functions,tolerance):
    actual=functions('native')(*values);expected=functions('warp')(*values)
    if isinstance(actual,tuple):
        torch.testing.assert_close(actual[1],expected[1],rtol=0,atol=0);actual,expected=actual[0],expected[0]
    seed=torch.linspace(.2,.7,actual.numel(),device=actual.device,dtype=actual.dtype).reshape_as(actual)
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
    got=torch.autograd.grad(actual,values,seed,allow_unused=True);want=torch.autograd.grad(expected,values,seed,allow_unused=True)
    for slot,(a,b,value) in enumerate(zip(got,want,values)):
        torch.testing.assert_close(torch.zeros_like(value) if a is None else a,torch.zeros_like(value) if b is None else b,
            rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda message:f'role {slot}: {message}')


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('corrected',[False,True])
def test_all_kernel_matrix_counts_and_reverse_products(device,kernel,dimension,corrected):
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,dimension,get_torch_precision(),kernel=kernel,
        support=list(sph.SupportScheme)[index%6],corrected=corrected,counts=True)
    compare(v,functions,1e-3 if v[0].dtype==torch.float32 else 5e-8)
    if not corrected:
        from warpSPHCore.coreOperations.wp_covarianceJVP import computeCovarianceGeometryJVP
        q,r,d=states(v[:10]);t=tuple(x.detach().cos()*.03 for x in v)
        options=dict(referenceParticles=r,queryTangentState=sph.ParticleTangentState(t[0],t[2],None,None),
            referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],t[9]))
        expected=computeCovarianceGeometryJVP(q,d,p.kernel,p.supportMode,g,operationMode=p.operationMode,**options)
        actual=native_jvp(q,p,d,adjacency=g,**options)
        torch.testing.assert_close(actual,expected,rtol=1e-3 if v[0].dtype==torch.float32 else 5e-8,atol=1e-3 if v[0].dtype==torch.float32 else 5e-8,equal_nan=True)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_support_and_direction_contracts(device,support,direction):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),support=support,counts=True)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),1e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('separation',['self','ultratiny','tiny','clamped','epsilon','near','cutoff','interior'])
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('corrected',[False,True])
def test_boundary_reverse_and_jvp_products(device,kernel,separation,dimension,corrected):
    from warpSPHCore.coreOperations.wp_covarianceJVP import computeCovarianceGeometryJVP
    from warpSPHCore.native._inputs import prepare_operator_inputs
    from warpSPHCore.native._build import build
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'tiny':eps*1e-5,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'interior':.56}[separation]
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,support=sph.SupportScheme.PartialSymmetric)
    qx=v[0].new_zeros((2,dimension));qx[:,0]=delta;rx=torch.zeros_like(qx);h=qx.new_full((2,),.8);mass=h.new_ones(2)
    kinds=torch.zeros(2,device=device,dtype=torch.int32);q=sph.ParticleState(qx,h,mass,kinds,mass);r=replace(q,positions=rx,supports=h.clone())
    d=replace(d,periodic=torch.zeros(dimension,device=device,dtype=torch.bool))
    g=sph.AdjacencyList(None,torch.arange(2,device=device,dtype=sph.get_torch_index_dtype()),torch.ones(2,device=device,dtype=torch.int32),torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype()),2,2)
    values=tuple(x.detach().clone().requires_grad_() for x in (qx,rx,h,h.clone(),mass,d.min,d.max))
    A=h.new_full((2,),.91);B=qx.new_full((2,dimension),.07);ga=qx.new_full((2,dimension),.03);gb=qx.new_full((2,dimension,dimension),-.02)
    renorm=torch.eye(dimension,device=device,dtype=dtype).repeat(2,1,1)*1.03
    def run(backend):
        def evaluate(a,b,c,e,m,lo,hi):
            options=dict(referenceParticles=replace(r,positions=b,supports=e,masses=m),adjacency=g,backend=backend)
            if corrected:options.update(crkState=sph.CRKState(A,B,ga,gb),renormalizationState=renorm)
            return sphOperation(replace(q,positions=a,supports=c),p,replace(d,min=lo,max=hi),**options)
        return evaluate
    tolerance=4e-3 if dtype==torch.float32 else 3e-7
    compare(values,run,tolerance)
    a,b,c,e,m,lo,hi=values;qi=replace(q,positions=a,supports=c);rj=replace(r,positions=b,supports=e,masses=m)
    if not corrected:
        packed=prepare_operator_inputs(qi,rj,replace(d,min=lo,max=hi),p,g,fields=(mass.new_empty(0),mass.new_empty(0),mass))
        slots=(0,1,2,3,4,7,8);seed=torch.full_like(run('native')(*values),.3)
        raw=build().covariance_adjoint(packed,seed,sum(1<<s for s in slots),p.supportMode.value,p.operationMode.value,p.kernel.value,p.gradientMode.value,2,False)
        owned=torch.autograd.grad(run('native')(*values),values,seed)
        for slot,want in zip(slots,owned):torch.testing.assert_close(raw[slot],want,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda message:f'raw role {slot}: {message}')
        qt=sph.ParticleTangentState(a.detach().new_full(a.shape,.02),c.detach()*.03,None,None)
        rt=sph.ParticleTangentState(b.detach().new_full(b.shape,-.01),e.detach()*-.02,m.detach()*.04,None)
        qi=replace(qi,positions=a.detach(),supports=c.detach());rj=replace(rj,positions=b.detach(),supports=e.detach(),masses=m.detach())
        options=dict(queryTangentState=qt,referenceTangentState=rt,referenceParticles=rj)
        actual=native_jvp(qi,p,d,adjacency=g,**options)
        expected=computeCovarianceGeometryJVP(qi,d,kernel,p.supportMode,g,operationMode=p.operationMode,**options)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
    else:
        # The public covariance JVP has no correction bundle. Compose the
        # supported scalar Gradient JVP with C=-outer(dx,V*G) per edge.
        from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP
        qi=replace(qi,positions=a.detach(),supports=c.detach());rj=replace(rj,positions=b.detach(),supports=e.detach(),masses=m.detach())
        qt=sph.ParticleTangentState(a.detach().new_full(a.shape,.02),c.detach()*.03,None,None)
        rt=sph.ParticleTangentState(b.detach().new_full(b.shape,-.01),e.detach()*-.02,m.detach()*.04,None)
        corrections=dict(crkState=sph.CRKState(A,B,ga,gb),renormalizationState=sph.RenormalizationState(renorm))
        options=dict(referenceParticles=rj,queryTangentState=qt,referenceTangentState=rt,**corrections)
        actual=native_jvp(qi,p,d,adjacency=g,**options)
        fields=dict(queryValues=torch.zeros_like(m),referenceValues=torch.ones_like(m))
        gradient_properties=replace(p,operation=sph.WarpOperation.Gradient,gradientMode=sph.GradientScheme.Naive)
        gradient=sphOperation(qi,gradient_properties,d,adjacency=g,referenceParticles=rj,backend='warp',**fields,**corrections)
        derivative=computeSPHGradientGeometryJVP(qi,d,kernel,p.supportMode,g,gradientMode=sph.GradientScheme.Naive,
            operationMode=p.operationMode,**fields,**options)
        dx=a.detach()-b.detach();direction=qt.positions-rt.positions
        expected=-direction[:,:,None]*gradient[:,None,:]-dx[:,:,None]*derivative[:,None,:]
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)


def test_calibration_grid_and_first_order_scope(device,monkeypatch):
    from warpSPHCore.radiusSearch.compactHash.buildHashmap import buildCompactHashMap
    dtype=get_torch_precision();q,r,d,p,g,v,states,functions=case(device,dtype=dtype,corrected=True)
    p=replace(p,calibrateNormalization=True,n_h=2.57)
    grid=buildCompactHashMap(q.positions,r.positions,q.supports,r.supports,d.periodic,d,p.supportMode)
    run=lambda backend:functions(backend,p,grid)
    expected=run('warp')(*v);seed=torch.ones_like(expected)
    want=torch.autograd.grad(expected,v,seed,create_graph=True,allow_unused=True)
    assert all(z is None or not z.requires_grad for z in want)
    with pytest.raises(RuntimeError,match='setup_context|storage'):torch.func.vjp(lambda x:run('warp')(x,*v[1:]),v[0])
    monkeypatch.setattr(wp,'launch',lambda *a,**k:pytest.fail('native covariance imported/launched Warp'))
    actual=run('native')(*v);got=torch.autograd.grad(actual,v,seed,allow_unused=True)
    tolerance=7e-4 if dtype==torch.float32 else 3e-8
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    for a,b,x in zip(got,want,v):torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('operation',[sph.WarpOperation.Covariance])
@pytest.mark.parametrize('separate_query_volume',[False,True])
def test_query_volume_fallback_and_unread_volume(device,operation,separate_query_volume):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision())
    p=replace(p,operation=operation)
    unread=v[13].new_ones(4).requires_grad_();values=(*v,unread)
    def function(backend):
        def run(*x):
            qi,rj,domain=states(x[:10])
            volume=x[-1] if separate_query_volume else x[13]
            return sphOperation(qi,p,domain,x[14],x[10],referenceParticles=rj,adjacency=g,
                queryVolumes=volume,referenceVolumes=x[13] if separate_query_volume else None,backend=backend)
        return run
    compare(values,function,7e-4 if v[0].dtype==torch.float32 else 3e-8)
    qi,rj,domain=states(v[:10]);tangent=v[13].detach()*.03
    # Warp routes correction-only products through its geometry dispatcher.
    options=dict(referenceParticles=rj,adjacency=g,referenceValues=v[10],
        queryTangentState=sph.ParticleTangentState(torch.zeros_like(qi.positions),None,None,None),
        queryValues=None,
        queryVolumes=unread if separate_query_volume else v[13],
        referenceVolumes=v[13] if separate_query_volume else None,
        tangentQueryVolumes=torch.ones_like(unread)*.07 if separate_query_volume else tangent,
        tangentReferenceVolumes=tangent if separate_query_volume else None)
    tolerance=7e-4 if v[0].dtype==torch.float32 else 3e-8
    actual=native_jvp(qi,p,domain,**options)
    # The reference JVP rejects query-volume roles; normalize their alias
    # to its supported reference-volume input for the numerical comparison.
    reference_options={key:value for key,value in options.items() if key not in ('queryVolumes','tangentQueryVolumes')}
    reference_options.update(referenceVolumes=v[13],tangentReferenceVolumes=tangent)
    expected=sph.warpOperationJVP(qi,p,domain,**reference_options)
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(actual,function('native')(*values)*.03,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('bundle',['state','tuple','tensor'])
def test_sole_unused_grad_h_zero_products(device,bundle):
    from warpSPHCore.coreOperations.native.wrapper import sphOperation
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision())
    q,r,d=states(tuple(x.detach() for x in v[:10]))
    omega=v[19].detach().clone().requires_grad_()
    def evaluate(value,backend):
        correction=sph.GradHState(value,value) if bundle=='state' else (value,value) if bundle=='tuple' else value
        return sphOperation(q,p,d,referenceParticles=r,adjacency=g,gradHState=correction,backend=backend)
    actual=evaluate(omega,'native');expected=evaluate(omega,'warp')
    tolerance=7e-4 if omega.dtype==torch.float32 else 3e-8
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    for output in (actual,expected):
        torch.testing.assert_close(torch.autograd.grad(output.sum(),omega)[0],torch.zeros_like(omega))
