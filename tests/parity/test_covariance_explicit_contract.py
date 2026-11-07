"""Covariance full-product reverse against the plain Warp geometry contract."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.covariance_explicit_contract import case,boundary_case
from _support.gradient_explicit_contract import DIRECTION_SLOTS


@pytest.fixture(scope='module',autouse=True)
def reference_runtime():wp.init()


def compare(v,functions,tolerance,reference_backend="warp",roundoff=None):
    for weighted in (False,True):
        actual=functions('native')(*v);expected=functions(reference_backend)(*v)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
        seed=torch.linspace(-.3,.7,actual.numel(),device=actual.device,dtype=actual.dtype).reshape_as(actual) if weighted else torch.ones_like(actual)
        slots=[k for k,x in enumerate(v) if x is not None and x.requires_grad];inputs=[v[k] for k in slots]
        bars=torch.autograd.grad(actual,inputs,seed,allow_unused=True);wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        budgets={} if roundoff is None else roundoff(seed)
        for k,a,b,x in zip(slots,bars,wanted,inputs):
            a=torch.zeros_like(x) if a is None else a;b=torch.zeros_like(x) if b is None else b
            if k not in budgets:
                torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'covariance role {k}: {msg}')
            else:
                finite=torch.isfinite(a)&torch.isfinite(b)
                torch.testing.assert_close(a[~finite],b[~finite],rtol=tolerance,atol=tolerance,equal_nan=True)
                allowance=tolerance+tolerance*b.abs()+budgets[k]
                assert bool(((a-b).abs()[finite]<=allowance[finite]).all()),f'covariance role {k} exceeds expanded-path roundoff bound'



@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('volume',[False,True])
def test_full_product_reverse_reference(device,kernel,volume):
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[i%6],volume=volume)
    compare(v,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('volume',[False,True])
def test_warp_full_product_actual_reverse_gradcheck(device,kernel,volume):
    if get_torch_precision()!=torch.float64:pytest.skip('finite differences use FP64')
    i=list(sph.KernelFunctions).index(kernel)
    *_,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[i%6],volume=volume)
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=8e-5,rtol=2e-3)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_boundary_product_reverse_support_ties(device,kernel,dimension,support,separation):
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta)
    from _support.covariance_roundoff import boundary_roundoff_budgets
    compare(v,functions,8e-3 if dtype==torch.float32 else 6e-7,roundoff=lambda seed:boundary_roundoff_budgets(v,kernel,support,seed))


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
@pytest.mark.parametrize('volume',[False,True])
def test_support_direction_and_weight_couplings(device,support,direction,volume):
    from dataclasses import replace
    i=list(sph.OperationDirection).index(direction)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),support=support,volume=volume);p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('role',['query_mass','query_density','reference_mass','reference_density','lower','upper','volume'])
def test_geometry_ignored_and_frozen_metadata(device,volume,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),volume=volume)
    slot={'query_mass':7,'query_density':8,'reference_mass':4,'reference_density':9,'lower':5,'upper':6,'volume':13}[role]
    values=[x.detach() for x in v];values[slot].requires_grad_();outputs=[functions(b)(*values) for b in ('native','warp')]
    expected=role not in ('lower','upper') and (role!='volume' or volume)
    assert all(y.requires_grad==expected for y in outputs)
    if expected:
        bars=[torch.autograd.grad(y.sum(),values[slot])[0] for y in outputs]
        torch.testing.assert_close(*bars,rtol=6e-3 if v[0].dtype==torch.float32 else 8e-8,atol=6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('role',['lower','upper'])
def test_geometry_ignores_frozen_forward_metadata(device,role):
    *_,v,states,functions=case(device,dtype=get_torch_precision());slot=5 if role=='lower' else 6
    for backend in ('native','warp'):
        with torch.autograd.forward_ad.dual_level():
            values=[x.detach() for x in v];values[slot]=torch.autograd.forward_ad.make_dual(values[slot],torch.ones_like(values[slot]));y=functions(backend)(*values)
            assert torch.autograd.forward_ad.unpack_dual(y).tangent is None


@pytest.mark.parametrize('role',['query_density','reference_density'])
def test_absent_unused_density_retains_direction_zero_dependency(device,role):
    *_,v,states,functions=case(device,dtype=get_torch_precision(),volume=True)
    slot,tangent=(8,25) if role=='query_density' else (9,26)
    values=[x.detach() for x in v];values[slot]=None;values[tangent]=values[tangent].clone().requires_grad_()
    for backend in ('native','warp'):
        y=functions(backend)(*values);assert y.requires_grad
        torch.testing.assert_close(torch.autograd.grad(y.sum(),values[tangent])[0],torch.zeros_like(values[tangent]))


@pytest.mark.parametrize('role',['query_position','reference_position','volume','query_density'])
def test_explicit_product_rejects_second_forward_level(device,role):
    *_,v,states,functions=case(device,dtype=get_torch_precision(),volume=True)
    slot={'query_position':0,'reference_position':1,'volume':13,'query_density':8}[role]
    for backend in ('native','warp'):
        with torch.autograd.forward_ad.dual_level():
            values=[x.detach() for x in v];values[slot]=torch.autograd.forward_ad.make_dual(values[slot],torch.ones_like(values[slot]))
            with pytest.raises((NotImplementedError,RuntimeError)):functions(backend)(*values)


def test_explicit_product_reference_higher_reverse_boundary(device):
    *_,v,states,functions=case(device,dtype=get_torch_precision(),volume=True)
    for backend in ('native','warp'):
        y=functions(backend)(*v);bars=torch.autograd.grad(y,[x for x in v if x.requires_grad],torch.ones_like(y),create_graph=True,allow_unused=True)
        assert all(x is None or not x.requires_grad for x in bars)


def test_public_volume_only_dispatch_and_reverse(device):
    from warpSPHCore.native.api import warpOperationJVP as native
    q,r,d,p,g,v,states,_=case(device,dtype=get_torch_precision(),volume=True)
    values=(*v,(v[13].detach().cos()*.03).requires_grad_())
    def functions(backend):
        def run(*x):
            qi,rj,domain=states(x[:10])
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,referenceParticles=rj,adjacency=g,referenceVolumes=x[13],tangentReferenceVolumes=x[-1])
        return run
    compare(values,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('correction',['crk','renorm','combined'])
def test_corrected_matrix_product_against_warp_gradient_identity(device,kernel,correction):
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[i%6],volume=correction=='combined',crk=correction in ('crk','combined'),renorm=correction in ('renorm','combined'))
    d.periodic.zero_()
    compare(v,functions,6e-3 if v[0].dtype==torch.float32 else 8e-8,'warp_gradient')


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_corrected_boundary_matrix_product_against_warp_formula(device,kernel,dimension,support,separation):
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta,True)
    from _support.covariance_roundoff import boundary_roundoff_budgets
    compare(v,functions,8e-3 if dtype==torch.float32 else 6e-7,'warp_extension',lambda seed:boundary_roundoff_budgets(v,kernel,support,seed,True))


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('separation',['clamped','epsilon','near','interior'])
def test_spiky_geometry_cancellation_against_high_precision_oracle(device,dimension,separation):
    from decimal import Decimal
    from _support.covariance_roundoff import boundary_roundoff_budgets,decimal_oracle
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'clamped':eps*.85,'epsilon':eps,'near':eps*10,'interior':.56}[separation]
    kernel=sph.KernelFunctions.Spiky;support=sph.SupportScheme.Gather
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta)
    for weighted in (False,True):
        seed=torch.linspace(-.3,.7,2*dimension*dimension,device=device,dtype=dtype).reshape(2,dimension,dimension) if weighted else torch.ones((2,dimension,dimension),device=device,dtype=dtype)
        expected=decimal_oracle(v,seed,'spiky',support);budgets=boundary_roundoff_budgets(v,kernel,support,seed)
        for backend in ('native','warp'):
            y=functions(backend)(*v);slots=tuple(expected);bars=torch.autograd.grad(y,[v[k] for k in slots],seed)
            for slot,bar in zip(slots,bars):
                assert torch.isfinite(bar).all()
                for actual,target,bound in zip(bar.flatten().tolist(),expected[slot],(budgets[slot]/2).flatten().tolist()):
                    assert abs(Decimal.from_float(actual)-target)<=Decimal.from_float(bound),(backend,slot,separation,actual,str(target),bound)


@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('separation',['epsilon','near','interior'])
def test_coefficient_cancellation_against_high_precision_oracle(device,support,separation):
    from decimal import Decimal
    from _support.covariance_roundoff import boundary_roundoff_budgets,decimal_oracle
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'epsilon':eps,'near':eps*10,'interior':.56}[separation]
    kernel=sph.KernelFunctions.ViscosityKernel;v,functions=boundary_case(device,1,dtype,kernel,support,delta,True)
    for weighted in (False,True):
        seed=v[0].new_tensor([[[-.3]],[[.7]]]) if weighted else torch.ones((2,1,1),device=device,dtype=dtype)
        expected=decimal_oracle(v,seed,'viscosity',support);budgets=boundary_roundoff_budgets(v,kernel,support,seed,True)
        for backend in ('native','warp_extension'):
            y=functions(backend)(*v);slots=tuple(expected);bars=torch.autograd.grad(y,[v[k] for k in slots],seed)
            for slot,bar in zip(slots,bars):
                assert torch.isfinite(bar).all()
                for actual,target,bound in zip(bar.flatten().tolist(),expected[slot],(budgets[slot]/2).flatten().tolist()):
                    assert abs(Decimal.from_float(actual)-target)<=Decimal.from_float(bound),(backend,slot,separation,actual,str(target),bound)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('correction',['crk','renorm','combined'])
def test_extension_reference_actual_reverse_gradcheck(device,kernel,correction):
    if get_torch_precision()!=torch.float64:pytest.skip('finite differences use FP64')
    i=list(sph.KernelFunctions).index(kernel)
    *_,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[i%6],volume=correction=='combined',crk=correction in ('crk','combined'),renorm=correction in ('renorm','combined'))
    assert torch.autograd.gradcheck(functions('warp_extension'),v,fast_mode=True,atol=8e-5,rtol=2e-3)
