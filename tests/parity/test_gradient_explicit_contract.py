"""Full gradient JVP reverse against the supported Warp geometry helper."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.gradient_explicit_contract import case


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
                torch.testing.assert_close(x,y,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'gradient product role {slot}: {msg}')
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
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,scheme=scheme,support=list(sph.SupportScheme)[i%6],**options)
    compare(v,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_boundary_product_reverse_support_ties_and_corrections(device,kernel,dimension,support,corrected,separation):
    from _support.gradient_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta,corrected)
    from _support.gradient_explicit_contract import coefficient_roundoff_budgets
    roundoff=(lambda seed:coefficient_roundoff_budgets(v,kernel,support,seed)) if corrected else None
    compare(v,functions,8e-3 if dtype==torch.float32 else 6e-7,roundoff)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('correction',['plain','volume','crk','renorm','combined'])
def test_warp_full_product_actual_reverse_gradcheck(device,kernel,correction):
    if get_torch_precision()!=torch.float64:pytest.skip('finite differences use FP64')
    i=list(sph.KernelFunctions).index(kernel);options=dict(volume=correction in ('volume','combined'),crk=correction in ('crk','combined'),renorm=correction in ('renorm','combined'))
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,scheme=list(sph.GradientScheme)[i%4],support=list(sph.SupportScheme)[i%6],**options)
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('field_shape',[(),(2,3,2)])
def test_field_only_reverse_retains_bounds_and_corrections(device,scheme,corrected,field_shape):
    from _support.gradient_contract import case as ordinary_case
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
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),support=support,scheme=list(sph.GradientScheme)[i%4],volume=True,crk=True,renorm=True)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('operation',[sph.WarpOperation.Interpolate,sph.WarpOperation.Gradient])
@pytest.mark.parametrize('bundle',['crk','renorm','both'])
def test_field_only_empty_correction_tangent_bundles(device,operation,bundle):
    from dataclasses import replace
    from _support.gradient_contract import case as ordinary_case
    from warpSPHCore.native.api import warpOperationJVP as native
    q,r,d,p,g,v,states,_=ordinary_case(device,dtype=get_torch_precision());p=replace(p,operation=operation)
    values=(*v,(v[14].detach().cos()*.03).requires_grad_(),(v[10].detach().cos()*.02).requires_grad_())
    def functions(backend):
        def run(*x):
            qi,rj,domain=states(x[:10]);options=dict(referenceParticles=rj,adjacency=g,queryValues=x[14],referenceValues=x[10],tangentQueryValues=x[20],tangentReferenceValues=x[21],referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],x[15],x[16]))
            if bundle in ('crk','both'):options['crkTangentState']=sph.CRKTangentState(None,None,None,None)
            if bundle in ('renorm','both'):options['renormalizationTangentState']=sph.RenormalizationTangentState(None)
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**options)
        return run
    compare(values,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('separation',['epsilon','near','interior'])
def test_coefficient_cancellation_against_high_precision_oracle(device,separation):
    from decimal import Decimal
    from _support.gradient_explicit_contract import boundary_case,coefficient_roundoff_budgets,viscosity_coefficient_decimal_oracle
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15;delta={'epsilon':eps,'near':eps*10,'interior':.56}[separation]
    v,functions=boundary_case(device,1,dtype,sph.KernelFunctions.ViscosityKernel,sph.SupportScheme.Gather,delta,True)
    for weighted in (False,True):
        seed=v[0].new_tensor([[-.3],[.7]]) if weighted else torch.ones_like(v[0]);expected=viscosity_coefficient_decimal_oracle(v,seed)
        budgets=coefficient_roundoff_budgets(v,sph.KernelFunctions.ViscosityKernel,sph.SupportScheme.Gather,seed)
        for backend in ('native','warp'):
            y=functions(backend)(*v);bars=torch.autograd.grad(y,(v[16],v[29]),seed)
            for slot,bar in zip((16,29),bars):
                assert torch.isfinite(bar).all()
                for actual,target,bound in zip(bar.flatten().tolist(),expected[slot],(budgets[slot]/2).flatten().tolist()):
                    assert abs(Decimal.from_float(actual)-target)<=Decimal.from_float(bound),(backend,slot,separation,actual,str(target),bound)


@pytest.mark.parametrize('role',['query_density','reference_density'])
def test_absent_unused_density_retains_direction_zero_dependency(device,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),volume=True)
    slot,tangent=(8,25) if role=='query_density' else (9,26)
    values=[x.detach() for x in v];values[slot]=None;values[tangent]=values[tangent].clone().requires_grad_()
    for backend in ('native','warp'):
        y=functions(backend)(*values);assert y.requires_grad
        torch.testing.assert_close(torch.autograd.grad(y.sum(),values[tangent])[0],torch.zeros_like(values[tangent]))


def test_interpolation_absent_density_retains_direction_zero_dependency(device):
    from _support.interpolate_explicit_contract import case as interpolate_case
    q,r,d,p,g,v,states,functions=interpolate_case(device,dtype=get_torch_precision(),volume=True)
    values=[x.detach() for x in v];values[9]=None;values[19]=values[19].clone().requires_grad_()
    for backend in ('native','warp'):
        y=functions(backend)(*values);assert y.requires_grad
        torch.testing.assert_close(torch.autograd.grad(y.sum(),values[19])[0],torch.zeros_like(values[19]))
