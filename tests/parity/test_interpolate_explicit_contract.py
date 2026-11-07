"""Interpolation geometry-product reverse versus the direct Warp helper."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.interpolate_explicit_contract import case


@pytest.fixture(scope='module',autouse=True)
def reference_runtime():wp.init()


def compare(values,functions,tolerance):
    for weighted in (False,True):
        actual=functions('native')(*values);expected=functions('warp')(*values)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
        seed=torch.linspace(-.3,.7,actual.numel(),device=actual.device,dtype=actual.dtype).reshape_as(actual) if weighted else torch.ones_like(actual)
        inputs=[x for x in values if x.requires_grad]
        a=torch.autograd.grad(actual,inputs,seed,allow_unused=True);b=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for slot,(x,y,z) in enumerate(zip(a,b,inputs)):
            torch.testing.assert_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,
                rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'product reverse role {slot}: {msg}')


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('volume',[False,True])
def test_all_kernel_full_product_reverse(device,kernel,dimension,volume):
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,dimension,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[i%6],volume=volume)
    compare(v,functions,3e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('direction',list(sph.OperationDirection))
@pytest.mark.parametrize('volume',[False,True])
def test_support_direction_and_weight_couplings(device,support,direction,volume):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),support=support,volume=volume)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),3e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('field_shape',[(),(5,),(2,3,2)])
def test_crk_geometry_extension_against_composed_reference(device,kernel,volume,field_shape):
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,volume=volume,crk=True,field_shape=field_shape)
    compare(v,functions,3e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('support',[sph.SupportScheme.Gather,sph.SupportScheme.PartialSymmetric,sph.SupportScheme.SuperSymmetric])
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_boundary_product_reverse_and_support_ties(device,kernel,dimension,support,volume,separation):
    from _support.interpolate_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    v,functions=boundary_case(device,dimension,dtype,kernel,support,delta,volume)
    compare(v,functions,4e-3 if dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('separation',['self','clamped','cutoff','interior'])
def test_crk_extension_boundary_composition(device,kernel,volume,separation):
    from _support.interpolate_explicit_contract import boundary_case
    dtype=get_torch_precision();eps=1e-7 if dtype==torch.float32 else 1e-15
    delta={'self':0.,'clamped':eps*.85,'cutoff':.8,'interior':.56}[separation]
    v,functions=boundary_case(device,1,dtype,kernel,sph.SupportScheme.Gather,delta,volume,True)
    compare(v,functions,4e-3 if dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('crk',[False,True])
@pytest.mark.parametrize('query_fallback',[False,True])
@pytest.mark.parametrize('field_shape',[(),(2,3,2)])
def test_field_only_reverse_bounds_crk_and_query_fallback(device,volume,crk,query_fallback,field_shape):
    from _support.interpolate_contract import case as ordinary_case
    from warpSPHCore.native.api import warpOperationJVP as native
    q,r,d,p,g,v,states,_=ordinary_case(device,dtype=get_torch_precision(),field_shape=field_shape)
    tf=(v[10].detach().cos()*.03).requires_grad_();values=(*v,tf)
    self_g=sph.AdjacencyList(None,torch.tensor([0,1,0,1],device=device,dtype=sph.get_torch_index_dtype()),torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device,dtype=sph.get_torch_offset_dtype()),2,2)
    def functions(backend):
        def run(*x):
            qi,rj,domain=states(x[:10]);self_case=query_fallback
            field=x[10][:2] if self_case else x[10];direction=x[14][:2] if self_case else x[14]
            options=dict(referenceParticles=qi if self_case else rj,adjacency=self_g if self_case else g,referenceValues=field)
            options['tangentQueryValues' if query_fallback else 'tangentReferenceValues']=direction
            if volume:options['referenceVolumes']=x[13][:2] if self_case else x[13]
            if crk:options['crkState']=sph.CRKState(x[11],x[12],None,None)
            return (native if backend=='native' else sph.warpOperationJVP)(qi,p,domain,**options)
        return run
    compare(values,functions,3e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('volume',[False,True])
@pytest.mark.parametrize('role',['query_mass','query_density','reference_mass','reference_density','lower','upper','volume'])
def test_geometry_product_ignored_and_frozen_metadata(device,volume,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision(),volume=volume)
    slot={'query_mass':7,'query_density':8,'reference_mass':4,'reference_density':9,'lower':5,'upper':6,'volume':13}[role]
    values=[x.detach() for x in v];values[slot].requires_grad_()
    outputs=[functions(b)(*values) for b in ('native','warp')]
    expected=role not in ('query_density','lower','upper') and (role!='volume' or volume)
    assert all(y.requires_grad==expected for y in outputs)
    if expected:
        gradients=[torch.autograd.grad(y.sum(),values[slot])[0] for y in outputs]
        torch.testing.assert_close(*gradients,rtol=3e-3 if v[0].dtype==torch.float32 else 5e-8,atol=3e-3 if v[0].dtype==torch.float32 else 5e-8)


@pytest.mark.parametrize('role',['query_density','lower','upper'])
def test_geometry_product_ignores_unused_forward_metadata(device,role):
    q,r,d,p,g,v,states,functions=case(device,dtype=get_torch_precision())
    slot={'query_density':8,'lower':5,'upper':6}[role]
    for backend in ('native','warp'):
        with torch.autograd.forward_ad.dual_level():
            values=[x.detach() for x in v];values[slot]=torch.autograd.forward_ad.make_dual(values[slot],torch.ones_like(values[slot]))
            y=functions(backend)(*values)
            assert torch.autograd.forward_ad.unpack_dual(y).tangent is None

@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('volume',[False,True])
def test_warp_geometry_product_actual_reverse_gradcheck(device,kernel,volume):
    if get_torch_precision()!=torch.float64:pytest.skip('finite differences use FP64')
    i=list(sph.KernelFunctions).index(kernel)
    q,r,d,p,g,v,states,functions=case(device,1+i%3,get_torch_precision(),kernel=kernel,support=list(sph.SupportScheme)[i%6],volume=volume)
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=6e-5,rtol=1e-3)
