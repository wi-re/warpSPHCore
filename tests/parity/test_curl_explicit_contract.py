"""Full gradient JVP reverse against the supported Warp geometry helper."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.curl_explicit_contract import case


@pytest.fixture(scope='module',autouse=True)
def reference_runtime():wp.init()


def compare(v,functions,tolerance,roundoff=None):
    for weighted in (False,True):
        actual=functions('native')(*v);expected=functions('warp')(*v)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
        seed=torch.linspace(-.3,.7,actual.numel(),device=actual.device,dtype=actual.dtype).reshape_as(actual) if weighted else torch.ones_like(actual)
        slots=[i for i,x in enumerate(v) if isinstance(x,torch.Tensor) and x.requires_grad];inputs=[v[i] for i in slots]
        a=torch.autograd.grad(actual,inputs,seed,allow_unused=True);b=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        budgets={} if roundoff is None else roundoff(seed)
        for slot,x,y,z in zip(slots,a,b,inputs):
            x=torch.zeros_like(z) if x is None else x;y=torch.zeros_like(z) if y is None else y
            if slot not in budgets:
                torch.testing.assert_close(x,y,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'curl product role {slot}: {msg}')
            else:
                finite=torch.isfinite(x)&torch.isfinite(y)
                torch.testing.assert_close(x[~finite],y[~finite],rtol=tolerance,atol=tolerance,equal_nan=True)
                allowance=tolerance+tolerance*y.abs()+budgets[slot]
                assert bool(((x-y).abs()[finite]<=allowance[finite]).all()),f'coefficient role {slot} exceeds expanded-path roundoff bound'



@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('correction',['plain','volume','crk','renorm','combined'])
def test_full_product_reverse_reference(device,kernel,scheme,correction):
    i=list(sph.KernelFunctions).index(kernel);options=dict(volume=correction in ('volume','combined'),crk=correction in ('crk','combined'),renorm=correction in ('renorm','combined'))
    q,r,d,p,g,v,states,functions=case(device,2,get_torch_precision(),kernel=kernel,scheme=scheme,support=list(sph.SupportScheme)[i%6],**options)
    compare(v,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[2])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_boundary_product_reverse_support_ties_and_corrections(device,kernel,dimension,support,corrected,separation):
    from _support.curl_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta,corrected)
    from _support.curl_explicit_contract import coefficient_roundoff_budgets
    roundoff=(lambda seed:coefficient_roundoff_budgets(v,kernel,support,seed)) if corrected else None
    compare(v,functions,8e-3 if dtype==torch.float32 else 6e-7,roundoff)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('correction',['plain','volume','crk','renorm','combined'])
def test_warp_full_product_actual_reverse_gradcheck(device,kernel,correction):
    if get_torch_precision()!=torch.float64:pytest.skip('finite differences use FP64')
    i=list(sph.KernelFunctions).index(kernel);options=dict(volume=correction in ('volume','combined'),crk=correction in ('crk','combined'),renorm=correction in ('renorm','combined'))
    q,r,d,p,g,v,states,functions=case(device,2,get_torch_precision(),kernel=kernel,scheme=list(sph.GradientScheme)[i%4],support=list(sph.SupportScheme)[i%6],**options)
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('field_shape',[(2,),(2,3,2)])
def test_field_only_reverse_retains_bounds_and_corrections(device,scheme,corrected,field_shape):
    from _support.curl_contract import case as ordinary_case
    from warpSPHCore.native.api import warpOperationJVP as native
    q,r,d,p,g,v,states,_=ordinary_case(device,dtype=get_torch_precision(),field_shape=field_shape,scheme=scheme)
    tq=(v[14].detach().cos()*.03).requires_grad_();tr=(v[10].detach().cos()*.02).requires_grad_();values=(*v,tq,tr)
    def functions(backend):
        def run(*x):
            qi,rj,domain=states(x[:10]);options=dict(referenceParticles=rj,adjacency=g,queryValues=x[14],referenceValues=x[10],tangentQueryValues=x[20],tangentReferenceValues=x[21])
            if corrected:options.update(referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],x[15],x[16]),gradHState=sph.GradHState(x[19],x[17]),renormalizationState=sph.RenormalizationState(x[18]))
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**options)
        return run
    compare(values,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('role',['query_mass','query_density','reference_mass','reference_density','lower','upper','volume'])
def test_geometry_ignored_and_frozen_metadata(device,scheme,volume,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),scheme=scheme,volume=volume)
    slot={'query_mass':7,'query_density':8,'reference_mass':4,'reference_density':9,'lower':5,'upper':6,'volume':13}[role]
    values=[x.detach() for x in v];values[slot].requires_grad_();outputs=[functions(b)(*values) for b in ('native','warp')]
    expected=role not in ('lower','upper') and (role!='volume' or volume)
    assert all(y.requires_grad==expected for y in outputs)
    if expected:
        bars=[torch.autograd.grad(y.sum(),values[slot])[0] for y in outputs]
        torch.testing.assert_close(*bars,rtol=6e-3 if v[0].dtype==torch.float32 else 8e-8,atol=6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('role',['lower','upper'])
def test_geometry_ignores_frozen_forward_metadata(device,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision());slot=5 if role=='lower' else 6
    for backend in ('native','warp'):
        with torch.autograd.forward_ad.dual_level():
            values=[x.detach() for x in v];values[slot]=torch.autograd.forward_ad.make_dual(values[slot],torch.ones_like(values[slot]));y=functions(backend)(*values)
            assert torch.autograd.forward_ad.unpack_dual(y).tangent is None


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_support_direction_and_weight_couplings(device,support,direction):
    from dataclasses import replace
    i=list(sph.OperationDirection).index(direction)
    q,r,d,p,g,v,states,functions=case(device,2,get_torch_precision(),support=support,scheme=list(sph.GradientScheme)[i%4],volume=True,crk=True,renorm=True)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('role',['query_density','reference_density'])
def test_absent_unused_density_retains_direction_zero_dependency(device,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),volume=True)
    slot,tangent=(8,25) if role=='query_density' else (9,26)
    values=[x.detach() for x in v];values[slot]=None;values[tangent]=values[tangent].clone().requires_grad_()
    for backend in ('native','warp'):
        y=functions(backend)(*values);assert y.requires_grad
        torch.testing.assert_close(torch.autograd.grad(y.sum(),values[tangent])[0],torch.zeros_like(values[tangent]))



@pytest.mark.parametrize('separation',['epsilon','near','interior'])
def test_coefficient_cancellation_against_high_precision_oracle(device,separation):
    from decimal import Decimal
    from _support.curl_explicit_contract import boundary_case,coefficient_roundoff_budgets,viscosity_coefficient_decimal_oracle
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15;delta={'epsilon':eps,'near':eps*10,'interior':.56}[separation]
    v,functions=boundary_case(device,2,dtype,sph.KernelFunctions.ViscosityKernel,sph.SupportScheme.Gather,delta,True)
    for weighted in (False,True):
        seed=v[0].new_tensor([[-.3],[.7]]) if weighted else torch.ones_like(v[2])[:,None]
        expected=viscosity_coefficient_decimal_oracle(v,seed);budgets=coefficient_roundoff_budgets(v,sph.KernelFunctions.ViscosityKernel,sph.SupportScheme.Gather,seed)
        for backend in ('native','warp'):
            y=functions(backend)(*v);bars=torch.autograd.grad(y,(v[16],v[29]),seed)
            for slot,bar in zip((16,29),bars):
                assert torch.isfinite(bar).all()
                for actual,target,bound in zip(bar.flatten().tolist(),expected[slot],(budgets[slot]/2).flatten().tolist()):
                    assert abs(Decimal.from_float(actual)-target)<=Decimal.from_float(bound),(backend,slot,separation,actual,str(target),bound)


def test_explicit_product_reference_higher_reverse_boundary(device):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),volume=True,crk=True,renorm=True)
    for backend in ('native','warp'):
        y=functions(backend)(*v)
        bars=torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True,create_graph=True)
        assert all(z is None or not z.requires_grad for z in bars)


@pytest.mark.parametrize('role',[0,20])
def test_explicit_product_rejects_second_forward_level(device,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision())
    for backend in ('native','warp'):
        with torch.autograd.forward_ad.dual_level():
            values=[x.detach() for x in v];values[role]=torch.autograd.forward_ad.make_dual(values[role],torch.ones_like(values[role]))
            with pytest.raises((RuntimeError,NotImplementedError,AssertionError)):
                functions(backend)(*values)




@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('bundle',['crk','renorm','both'])
def test_packed_field_only_empty_correction_tangent_bundles(device,dimension,scheme,bundle):
    from _support.curl_contract import case as ordinary_case
    from warpSPHCore.native.api import warpOperationJVP as native
    q,r,d,p,g,v,states,_=ordinary_case(device,dimension,get_torch_precision(),field_shape=(2,3,2) if dimension==2 else (dimension,2,3),scheme=scheme)
    values=(*v,(v[14].detach().cos()*.03).requires_grad_(),(v[10].detach().cos()*.02).requires_grad_())
    def functions(backend):
        def run(*x):
            qi,rj,domain=states(x[:10]);options=dict(referenceParticles=rj,adjacency=g,queryValues=x[14],referenceValues=x[10],tangentQueryValues=x[20],tangentReferenceValues=x[21],referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],x[15],x[16]),gradHState=sph.GradHState(x[19],x[17]),renormalizationState=sph.RenormalizationState(x[18]))
            if bundle in ('crk','both'):options['crkTangentState']=sph.CRKTangentState(None,None,None,None)
            if bundle in ('renorm','both'):options['renormalizationTangentState']=sph.RenormalizationTangentState(None)
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**options)
        return run
    compare(values,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('separation',['self','ultratiny','epsilon','near','cutoff','outside','interior'])
def test_three_dimensional_boundary_reverse_gradient_composition(device,kernel,separation):
    from _support.curl_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,3,dtype,kernel,sph.SupportScheme.PartialSymmetric,delta,False)
    v=list(v)
    # Nonparallel fields and geometry directions isolate the singular kernel
    # adjoint from the cross-product's separately checked zero identity.
    for slot,shift in ((10,[.03,-.11,.17]),(14,[-.07,.13,.29]),(27,[.02,.04,-.01]),(31,[-.03,.01,.05]),(20,[.01,-.02,.04]),(21,[-.02,.01,-.01])):
        v[slot]=(v[slot].detach()+v[slot].new_tensor(shift)).requires_grad_()
    v=tuple(v)
    compare(v,lambda backend:functions('native' if backend=='native' else 'warp3d'),8e-3 if dtype==torch.float32 else 6e-7)


@pytest.mark.parametrize('kernel',[sph.KernelFunctions.QuarticSpline,sph.KernelFunctions.QuinticSpline,sph.KernelFunctions.HOCT4,sph.KernelFunctions.Spiky])
@pytest.mark.parametrize('separation',['self','ultratiny','epsilon'])
def test_three_dimensional_zero_identity_roundoff(device,kernel,separation):
    from _support.curl_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15;delta={'self':0.,'ultratiny':eps*1e-9,'epsilon':eps}[separation]
    v,functions=boundary_case(device,3,dtype,kernel,sph.SupportScheme.PartialSymmetric,delta,False)
    v=list(v)
    for k in (10,14,27,31):v[k]=v[k].detach()[:,:1].expand(-1,3).clone().requires_grad_()
    v=tuple(v)
    slots=[k for k,x in enumerate(v) if x.requires_grad];inputs=[v[k] for k in slots];geometry=[k for k in slots if k not in (10,14,27,31)]
    tolerance=8e-3 if dtype==torch.float32 else 6e-7;machine=torch.finfo(dtype).eps;gamma=256*machine/(1-256*machine)
    # Equal field components imply sum(curl)==0 identically for every shared
    # geometry/weight/direction role. Test both backends against that exact
    # identity, with a condition bound from individual scalar-gradient axes.
    for weighted in (False,True):
        weight=v[2].new_tensor([-.3,.7]) if weighted else torch.ones_like(v[2])
        observed=[]
        for backend,component_backend in (('native','native_components'),('warp3d','warp_components')):
            y=functions(backend)(*v);seed=weight[:,None].expand_as(y)
            bars=torch.autograd.grad(y,inputs,seed,allow_unused=True)
            bars={k:torch.zeros_like(x) if a is None else a for k,x,a in zip(slots,inputs,bars)};observed.append(bars)
            components=functions(component_backend)(*v);condition={k:torch.zeros_like(v[k]) for k in geometry}
            for field in range(3):
                for axis in range(3):
                    basis=torch.zeros_like(components);basis[:,field,axis]=weight.abs()
                    partial=torch.autograd.grad(components,inputs,basis,allow_unused=True,retain_graph=True)
                    for k,a in zip(slots,partial):
                        if k in condition and a is not None:condition[k]+=a.abs()
            for k in geometry:
                finite=torch.isfinite(bars[k]);assert torch.isfinite(condition[k][finite]).all(),(backend,k,'nonfinite condition for finite identity residual')
                assert bool((bars[k].abs()[finite]<=(tolerance+gamma*condition[k])[finite]).all()),(backend,k,'zero identity exceeds expanded cross-product roundoff bound')
        for k in geometry:
            a,b=observed[0][k],observed[1][k];finite=torch.isfinite(a)&torch.isfinite(b)
            torch.testing.assert_close(a[~finite],b[~finite],rtol=tolerance,atol=tolerance,equal_nan=True)
