"""Gradient numerical scope versus Warp's taped reverse and explicit JVP paths."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.coreOperations.native.wrapper import sphOperation
from warpSPHCore.native.api import warpOperationJVP as native_jvp
from _support.gradient_contract import case


@pytest.fixture(scope='module',autouse=True)
def initialize_reference():
    wp.init()


def compare(values,functions,tolerance):
    actual=functions('native')(*values);expected=functions('warp')(*values)
    seed=actual.detach().cos().contiguous()
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
    got=torch.autograd.grad(actual,values,seed,allow_unused=True)
    want=torch.autograd.grad(expected,values,seed,allow_unused=True)
    for a,b,value in zip(got,want,values):
        torch.testing.assert_close(torch.zeros_like(value) if a is None else a,
            torch.zeros_like(value) if b is None else b,rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('corrected',[False,True])
def test_all_kernel_values_and_reverse_products(device,kernel,dimension,corrected):
    dtype=get_torch_precision();index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,scheme=list(sph.GradientScheme)[index%4],
                                   support=list(sph.SupportScheme)[index%6],corrected=corrected)
    tolerance=7e-4 if dtype==torch.float32 else 3e-8
    compare(v,functions,tolerance)
    if not corrected and p.supportMode!=sph.SupportScheme.PartialSymmetric:
        qi,rj,domain=states(tuple(x.detach() for x in v[:10]))
        t=tuple(.03*x.detach().cos() for x in v)
        options=dict(referenceParticles=rj,adjacency=g,referenceValues=v[10].detach(),
            queryValues=v[14].detach(),tangentQueryValues=t[14],tangentReferenceValues=t[10],
            queryTangentState=sph.ParticleTangentState(t[0],t[2],None,None),
            referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],t[9]))
        torch.testing.assert_close(native_jvp(qi,p,domain,**options),
            sph.warpOperationJVP(qi,p,domain,**options),rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_support_and_direction_contracts(device,support,direction):
    q,r,d,p,g,v,states,functions=case(device,2,get_torch_precision(),support=support)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),7e-4 if v[0].dtype==torch.float32 else 3e-8)




@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('separation',['self','tiny','epsilon','near','cutoff','interior'])
def test_boundary_products_and_registered_adjoint(device,kernel,separation):
    from warpSPHCore.native._inputs import prepare_operator_inputs
    from warpSPHCore.native._build import build
    from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'tiny':eps*1e-5,'epsilon':eps,'near':eps*10,'cutoff':.8,'interior':.56}[separation]
    q,r,d,p,g,v,states,functions=case(device,2,dtype,kernel=kernel,support=sph.SupportScheme.PartialSymmetric)
    qx=v[0].new_tensor([[delta,0],[delta,0]]);rx=torch.zeros_like(qx);h=qx.new_full((2,),.8)
    kinds=torch.zeros(2,device=device,dtype=torch.int32);mass=h.new_ones(2)
    q=sph.ParticleState(qx,h,mass,kinds,mass);r=replace(q,positions=rx,supports=h.clone())
    d=replace(d,periodic=torch.zeros(2,device=device,dtype=torch.bool))
    g=sph.AdjacencyList(None,torch.arange(2,device=device,dtype=sph.get_torch_index_dtype()),torch.ones(2,device=device,dtype=torch.int32),torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype()),2,2)
    values=tuple(x.detach().clone().requires_grad_() for x in (qx,rx,h,h.clone(),mass,d.min,d.max))
    fi=h.new_tensor([.2,-.3]);fj=h.new_tensor([.9,.4])
    def run(backend):
        return lambda a,b,c,e,m,lo,hi:sphOperation(replace(q,positions=a,supports=c),p,replace(d,min=lo,max=hi),fi,fj,
            referenceParticles=replace(r,positions=b,supports=e,masses=m),adjacency=g,backend=backend)
    compare(values,run,3e-3 if dtype==torch.float32 else 2e-7)
    a,b,c,e,m,lo,hi=values;qi=replace(q,positions=a,supports=c);rj=replace(r,positions=b,supports=e,masses=m)
    packed=prepare_operator_inputs(qi,rj,replace(d,min=lo,max=hi),p,g,fields=(fi,fj,mass))
    slots=(0,1,2,3,4,7,8);seed=a.new_tensor([[.2,.1],[-.6,.3]])
    raw=build().gradient_adjoint(packed,seed,sum(1<<s for s in slots),p.supportMode.value,p.operationMode.value,p.kernel.value,p.gradientMode.value,2,False)
    owned=torch.autograd.grad(run('native')(*values),values,seed)
    tolerance=3e-3 if dtype==torch.float32 else 2e-7
    for slot,want in zip(slots,owned):torch.testing.assert_close(raw[slot],want,rtol=tolerance,atol=tolerance,equal_nan=True)
    qt=sph.ParticleTangentState(a.detach().new_full(a.shape,.02),c.detach()*.03,None,None)
    rt=sph.ParticleTangentState(b.detach().new_full(b.shape,-.01),e.detach()*-.02,m.detach()*.04,None)
    qi=replace(qi,positions=a.detach(),supports=c.detach());rj=replace(rj,positions=b.detach(),supports=e.detach(),masses=m.detach())
    reference=computeSPHGradientGeometryJVP(qi,d,kernel,p.supportMode,g,qt,rj,rt,gradientMode=p.gradientMode,queryValues=fi,referenceValues=fj,operationMode=p.operationMode)
    actual=native_jvp(qi,p,d,queryTangentState=qt,referenceTangentState=rt,referenceParticles=rj,adjacency=g,queryValues=fi,referenceValues=fj)
    torch.testing.assert_close(actual,reference,rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('field_shape',[(5,),(2,5),(2,3,2)])
@pytest.mark.parametrize('corrected',[False,True])
def test_packed_field_shape_and_reverse_parity(device,field_shape,corrected):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),field_shape=field_shape,corrected=corrected)
    compare(v,functions,7e-4 if v[0].dtype==torch.float32 else 3e-8)


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
    monkeypatch.setattr(wp,'launch',lambda *a,**k:pytest.fail('native gradient imported/launched Warp'))
    actual=run('native')(*v);got=torch.autograd.grad(actual,v,seed,allow_unused=True)
    tolerance=7e-4 if dtype==torch.float32 else 3e-8
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
    for a,b,x in zip(got,want,v):torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_corrected_explicit_jvp_products(device,kernel):
    from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP
    index=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+index%3,get_torch_precision(),kernel=kernel,
        support=list(sph.SupportScheme)[index%6],scheme=list(sph.GradientScheme)[index%4])
    q,r,d=states(v[:10]);t=tuple(x.detach().cos()*.03 for x in v)
    options=dict(referenceParticles=r,referenceTangentState=sph.ParticleTangentState(t[1],t[3],t[4],t[9]),
        queryTangentState=sph.ParticleTangentState(t[0],t[2],None,t[8]),queryValues=v[14],referenceValues=v[10],
        tangentQueryValues=t[14],tangentReferenceValues=t[10],referenceVolumes=v[13],tangentReferenceVolumes=t[13],
        crkState=sph.CRKState(v[11],v[12],v[15],v[16]),crkTangentState=sph.CRKTangentState(t[11],t[12],t[15],t[16]),
        renormalizationState=sph.RenormalizationState(v[18]),renormalizationTangentState=sph.RenormalizationTangentState(t[18]))
    expected=computeSPHGradientGeometryJVP(q,d,p.kernel,p.supportMode,g,gradientMode=p.gradientMode,operationMode=p.operationMode,**options)
    actual=native_jvp(q,p,d,adjacency=g,**options)
    tolerance=8e-4 if v[0].dtype==torch.float32 else 4e-8
    torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)


@pytest.mark.parametrize('operation',[sph.WarpOperation.Interpolate,sph.WarpOperation.Gradient])
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
        queryValues=v[14] if operation==sph.WarpOperation.Gradient else None,
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
