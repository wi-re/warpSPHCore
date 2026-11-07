"""Full Laplacian JVP reverse against supported Warp geometry helpers."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.laplacian_explicit_contract import case,boundary_case


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
                torch.testing.assert_close(x,y,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'laplacian product role {slot}: {msg}')
            else:
                finite=torch.isfinite(x)&torch.isfinite(y)
                torch.testing.assert_close(x[~finite],y[~finite],rtol=tolerance,atol=tolerance,equal_nan=True)
                allowance=tolerance+tolerance*y.abs()+budgets[slot]
                assert bool(((x-y).abs()[finite]<=allowance[finite]).all()),f'boundary derivative role {slot} exceeds expanded-path roundoff bound'


CONFIGURATIONS=[(mode,correction) for mode in sph.LaplacianScheme for correction in ('plain','volume','crk','renorm','combined') if mode!=sph.LaplacianScheme.Naive or correction in ('plain','volume')]


@pytest.mark.parametrize('mode,correction',CONFIGURATIONS)
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
def test_full_product_reverse_reference(device,mode,correction,kernel,scheme):
    i=list(sph.KernelFunctions).index(kernel);dimension=1+i%3
    options=dict(volume=correction in ('volume','combined'),crk=correction in ('crk','combined'),renorm=correction in ('renorm','combined'))
    q,r,d,p,g,v,states,functions=case(device,dimension,get_torch_precision(),kernel=kernel,mode=mode,scheme=scheme,field_shape=(dimension,) if mode==sph.LaplacianScheme.Dot else (),support=list(sph.SupportScheme)[i%6],**options)
    compare(v,functions,1e-2 if v[0].dtype==torch.float32 else 2e-7)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_boundary_product_reverse_support_ties(device,mode,kernel,dimension,support,separation):
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta,mode=mode)
    compare(v,functions,2e-2 if dtype==torch.float32 else 2e-6)


@pytest.mark.parametrize('mode,correction',CONFIGURATIONS)
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_warp_full_product_actual_reverse_gradcheck(device,mode,correction,kernel):
    if get_torch_precision()!=torch.float64:pytest.skip('finite differences use FP64')
    i=list(sph.KernelFunctions).index(kernel);dimension=1+i%3
    options=dict(volume=correction in ('volume','combined'),crk=correction in ('crk','combined'),renorm=correction in ('renorm','combined'))
    q,r,d,p,g,v,states,functions=case(device,dimension,get_torch_precision(),kernel=kernel,mode=mode,field_shape=(dimension,) if mode==sph.LaplacianScheme.Dot else (),scheme=list(sph.GradientScheme)[i%4],support=list(sph.SupportScheme)[i%6],**options)
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=2e-4,rtol=3e-3)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('role',['query_mass','query_density','reference_mass','reference_density','lower','upper','volume'])
def test_geometry_ignored_and_frozen_metadata(device,mode,scheme,volume,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),mode=mode,field_shape=(2,) if mode==sph.LaplacianScheme.Dot else (),scheme=scheme,volume=volume)
    slot={'query_mass':7,'query_density':8,'reference_mass':4,'reference_density':9,'lower':5,'upper':6,'volume':13}[role]
    values=[x.detach() for x in v];values[slot].requires_grad_();outputs=[functions(b)(*values) for b in ('native','warp')]
    expected=role not in ('lower','upper') and (role!='volume' or volume)
    assert all(y.requires_grad==expected for y in outputs)
    if expected:
        bars=[torch.autograd.grad(y.sum(),values[slot])[0] for y in outputs]
        torch.testing.assert_close(*bars,rtol=6e-3 if v[0].dtype==torch.float32 else 8e-8,atol=6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('role',['lower','upper'])
def test_geometry_ignores_frozen_forward_metadata(device,mode,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),mode=mode,field_shape=(2,) if mode==sph.LaplacianScheme.Dot else ());slot=5 if role=='lower' else 6
    for backend in ('native','warp'):
        with torch.autograd.forward_ad.dual_level():
            values=[x.detach() for x in v];values[slot]=torch.autograd.forward_ad.make_dual(values[slot],torch.ones_like(values[slot]));y=functions(backend)(*values)
            assert torch.autograd.forward_ad.unpack_dual(y).tangent is None


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
def test_support_direction_and_weight_couplings(device,mode,support,direction):
    from dataclasses import replace
    i=list(sph.OperationDirection).index(direction)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),mode=mode,field_shape=(1+i%3,) if mode==sph.LaplacianScheme.Dot else (),support=support,scheme=list(sph.GradientScheme)[i%4],volume=True,crk=mode!=sph.LaplacianScheme.Naive,renorm=mode!=sph.LaplacianScheme.Naive)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),6e-3 if v[0].dtype==torch.float32 else 8e-8)




@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('positive',[False,True])
def test_field_only_reverse_retains_bounds_and_corrections(device,mode,scheme,corrected,packed,positive):
    from _support.laplacian_contract import case as ordinary_case
    from warpSPHCore.native.api import warpOperationJVP as native
    shape=(2,3,2) if packed else (2,) if mode==sph.LaplacianScheme.Dot or positive else ()
    q,r,d,p,g,v,states,_=ordinary_case(device,dtype=get_torch_precision(),mode=mode,positive=positive,field_shape=shape,scheme=scheme)
    values=(*v,(v[14].detach().cos()*.03).requires_grad_(),(v[10].detach().cos()*.02).requires_grad_())
    def functions(backend):
        def run(*x):
            qi,rj,domain=states(x[:10]);options=dict(referenceParticles=rj,adjacency=g,queryValues=x[14],referenceValues=x[10],tangentQueryValues=x[20],tangentReferenceValues=x[21])
            if corrected:options.update(referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],x[15],x[16]),gradHState=sph.GradHState(x[19],x[17]),renormalizationState=sph.RenormalizationState(x[18]))
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**options)
        return run
    if positive:
        # Warp's public field-only helper filters the supplied direction as a
        # fresh primal. That is not the derivative of the primal-mask branch.
        # Qualify the native extension against the ordinary Warp equation on
        # an independently frozen primal mask, including packed tails.
        from _support.positive_field_product_contract import case as positive_case
        values,product,reference,_=positive_case(device,mode,dtype=get_torch_precision(),corrected=corrected,scheme=scheme,field_shape=shape)
        functions=lambda backend:product if backend=='native' else lambda *x:reference('warp',*x)
    compare(values,functions,1e-2 if v[0].dtype==torch.float32 else 2e-7)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('bundle',['crk','renorm','both'])
def test_field_only_empty_correction_tangent_bundles(device,mode,bundle):
    from _support.laplacian_contract import case as ordinary_case
    from warpSPHCore.native.api import warpOperationJVP as native
    q,r,d,p,g,v,states,_=ordinary_case(device,dtype=get_torch_precision(),mode=mode,field_shape=(2,3,2))
    values=(*v,(v[14].detach().cos()*.03).requires_grad_(),(v[10].detach().cos()*.02).requires_grad_())
    def functions(backend):
        def run(*x):
            qi,rj,domain=states(x[:10]);options=dict(referenceParticles=rj,adjacency=g,queryValues=x[14],referenceValues=x[10],tangentQueryValues=x[20],tangentReferenceValues=x[21],referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],x[15],x[16]),renormalizationState=sph.RenormalizationState(x[18]))
            if bundle in ('crk','both'):options['crkTangentState']=sph.CRKTangentState(None,None,None,None)
            if bundle in ('renorm','both'):options['renormalizationTangentState']=sph.RenormalizationTangentState(None)
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**options)
        return run
    compare(values,functions,1e-2 if v[0].dtype==torch.float32 else 2e-7)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
def test_explicit_product_reference_higher_reverse_boundary(device,mode):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),mode=mode,field_shape=(2,) if mode==sph.LaplacianScheme.Dot else (),volume=True)
    for backend in ('native','warp'):
        y=functions(backend)(*v);bars=torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),allow_unused=True,create_graph=True)
        assert all(z is None or not z.requires_grad for z in bars)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('role',[0,20])
def test_explicit_product_rejects_second_forward_level(device,mode,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),mode=mode,field_shape=(2,) if mode==sph.LaplacianScheme.Dot else ())
    for backend in ('native','warp'):
        with torch.autograd.forward_ad.dual_level():
            values=[x.detach() for x in v];values[role]=torch.autograd.forward_ad.make_dual(values[role],torch.ones_like(values[role]))
            with pytest.raises((RuntimeError,NotImplementedError,AssertionError)):functions(backend)(*values)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('role',['query_density','reference_density'])
def test_absent_unused_density_retains_direction_zero_dependency(device,mode,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),mode=mode,field_shape=(2,) if mode==sph.LaplacianScheme.Dot else (),volume=True)
    slot,tangent=(8,25) if role=='query_density' else (9,26)
    values=[x.detach() for x in v];values[slot]=None;values[tangent]=values[tangent].clone().requires_grad_()
    for backend in ('native','warp'):
        y=functions(backend)(*values);assert y.requires_grad
        torch.testing.assert_close(torch.autograd.grad(y.sum(),values[tangent])[0],torch.zeros_like(values[tangent]))


@pytest.mark.parametrize('mode',[sph.LaplacianScheme.Brookshaw,sph.LaplacianScheme.Dot,sph.LaplacianScheme.Default])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_corrected_estimator_boundary_reverse(device,mode,kernel,dimension,support,separation):
    from _support.laplacian_explicit_contract import boundary_roundoff_budgets
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta,True,mode=mode)
    compare(v,functions,2e-2 if dtype==torch.float32 else 2e-6,lambda seed:boundary_roundoff_budgets(v,kernel,support,seed,mode))


@pytest.mark.parametrize('mode',[sph.LaplacianScheme.Brookshaw,sph.LaplacianScheme.Dot,sph.LaplacianScheme.Default])
@pytest.mark.parametrize('separation',['epsilon','near','interior'])
def test_coefficient_cancellation_against_high_precision_oracle(device,mode,separation):
    from decimal import Decimal
    from _support.laplacian_explicit_contract import boundary_case,boundary_roundoff_budgets,viscosity_coefficient_decimal_oracle
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15;delta={'epsilon':eps,'near':eps*10,'interior':.56}[separation]
    v,functions=boundary_case(device,1,dtype,sph.KernelFunctions.ViscosityKernel,sph.SupportScheme.Gather,delta,True,mode=mode)
    for weighted in (False,True):
        example=functions('native')(*v);seed=torch.linspace(-.3,.7,example.numel(),device=device,dtype=dtype).reshape_as(example) if weighted else torch.ones_like(example);expected=viscosity_coefficient_decimal_oracle(v,seed,mode)
        budgets=boundary_roundoff_budgets(v,sph.KernelFunctions.ViscosityKernel,sph.SupportScheme.Gather,seed,mode)
        for backend in ('native','warp'):
            y=functions(backend)(*v);slots=(11,12,15,16,18,28,29,32,33,34);bars=torch.autograd.grad(y,tuple(v[k] for k in slots),seed)
            for slot,bar in zip(slots,bars):
                assert torch.isfinite(bar).all()
                for actual,target,bound in zip(bar.flatten().tolist(),expected[slot],(budgets[slot]/2).flatten().tolist()):
                    assert abs(Decimal.from_float(actual)-target)<=Decimal.from_float(bound),(backend,slot,separation,actual,str(target),bound)


@pytest.mark.parametrize('kernel',[sph.KernelFunctions.CubicSpline,sph.KernelFunctions.ViscosityKernel])
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('separation',['clamped','epsilon','near','interior'])
def test_dot_direction_cancellation_against_high_precision_oracle(device,kernel,dimension,separation):
    from decimal import Decimal
    from _support.laplacian_explicit_contract import boundary_roundoff_budgets,dot_direction_decimal_oracle
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'clamped':eps*.85,'epsilon':eps,'near':eps*10,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,sph.SupportScheme.Gather,delta,True,mode=sph.LaplacianScheme.Dot)
    for weighted in (False,True):
        example=functions('native')(*v);seed=torch.linspace(-.3,.7,example.numel(),device=device,dtype=dtype).reshape_as(example) if weighted else torch.ones_like(example)
        expected=dot_direction_decimal_oracle(v,seed,kernel)
        bounds=boundary_roundoff_budgets(v,kernel,sph.SupportScheme.Gather,seed,sph.LaplacianScheme.Dot)
        for backend in ('native','warp'):
            y=functions(backend)(*v);bars=torch.autograd.grad(y,(v[20],v[21]),seed)
            for slot,bar in zip((20,21),bars):
                assert torch.isfinite(bar).all()
                for actual,target,bound in zip(bar.flatten().tolist(),expected[slot],(bounds[slot]/2).flatten().tolist()):
                    assert abs(Decimal.from_float(actual)-target)<=Decimal.from_float(bound),(backend,slot,dimension,kernel,separation,actual,str(target),bound)
