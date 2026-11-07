"""Ordinary and explicit polynomial products against readable Warp kernels."""
from dataclasses import replace
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.polyfit_contract import case


@pytest.fixture(scope='module',autouse=True)
def runtime():wp.init()


def bounded_close(a,b,tolerance,budget=None):
    torch.testing.assert_close(torch.isnan(a),torch.isnan(b))
    torch.testing.assert_close(torch.isposinf(a),torch.isposinf(b))
    torch.testing.assert_close(torch.isneginf(a),torch.isneginf(b))
    finite=torch.isfinite(a)&torch.isfinite(b)
    error=(a-b).abs();limit=tolerance*(1+b.abs())
    if budget is not None:limit=limit+budget
    assert torch.all(error[finite]<=limit[finite]), f'max excess {(error-limit)[finite].max().item()}'


def compare(v,functions,tolerance,roundoff=None):
    inputs=[x for x in v if x.requires_grad]
    for weighted in (False,True):
        a,b=functions('native')(*v),functions('warp')(*v)
        budget=None
        if roundoff is not None:
            exact,budget,_=roundoff(None)
            bounded_close(a,exact,tolerance,budget);bounded_close(b,exact,tolerance,budget)
        bounded_close(a,b,tolerance,budget)
        seed=torch.linspace(-.3,.7,a.numel(),device=a.device,dtype=a.dtype).reshape_as(a) if weighted else torch.ones_like(a)
        ga=torch.autograd.grad(a,inputs,seed,allow_unused=True);gb=torch.autograd.grad(b,inputs,seed,allow_unused=True)
        reverse=None if roundoff is None else roundoff(seed)[2]
        for k,(x,y,z) in enumerate(zip(ga,gb,inputs)):
            bounded_close(torch.zeros_like(z) if x is None else x,torch.zeros_like(z) if y is None else y,
                          tolerance,None if reverse is None else reverse[k])


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
def test_all_kernel_values_scalar_and_weighted_reverse(device,kernel,dimension,rhs,explicit):
    *_,v,functions=case(device,dimension,get_torch_precision(),kernel,rhs=rhs,explicit=explicit,subtract=rhs)
    compare(v,functions,9e-3 if v[0].dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('unit',[False,True])
def test_support_and_unit_weight_policies(device,support,rhs,explicit,unit):
    *_,v,functions=case(device,dtype=get_torch_precision(),support=support,rhs=rhs,explicit=explicit,subtract=rhs,unit=unit)
    compare(v,functions,9e-3 if v[0].dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_raw_boundary_values_and_reverse(device,kernel,dimension,rhs,explicit,separation):
    q,r,d,p,g,e,v,functions=case(device,dimension,get_torch_precision(),kernel,rhs=rhs,explicit=explicit,subtract=rhs)
    eps=1e-7 if v[0].dtype==torch.float32 else 1e-15
    delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
    d.periodic.zero_();v=list(v)
    if explicit:
        v[0]=torch.zeros_like(v[0]);v[0][::2,0]=delta;v[1]=torch.full_like(v[1],.8)
        g.paddedJ=torch.tensor([1,0,3,2],device=device,dtype=sph.get_torch_index_dtype())
        g.numNeighbors=torch.ones(4,device=device,dtype=torch.int32);g.edgeOffsets=torch.arange(4,device=device,dtype=sph.get_torch_offset_dtype())
    else:
        v[0]=torch.zeros_like(v[0]);v[0][:,0]=delta;v[1]=torch.zeros_like(v[1]);v[2]=torch.full_like(v[2],.8);v[3]=torch.full_like(v[3],.8)
        g.paddedJ=torch.arange(2,device=device,dtype=sph.get_torch_index_dtype())
        g.numNeighbors=torch.ones(2,device=device,dtype=torch.int32);g.edgeOffsets=torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype())
    v=tuple(x.detach().requires_grad_() for x in v)
    roundoff=None
    if kernel==sph.KernelFunctions.ViscosityKernel and separation in ('ultratiny','clamped','epsilon','near'):
        from _support.polyfit_roundoff import viscosity_fixture
        roundoff=lambda seed:viscosity_fixture(v,e,g,kernel,rhs=rhs,explicit=explicit,seed=seed)
    compare(v,functions,2e-2 if v[0].dtype==torch.float32 else 3e-6,roundoff)


@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('unit',[False,True])
def test_calibration_and_ignored_positive_flag(device,rhs,unit):
    q,r,d,p,g,e,v,functions=case(device,dtype=get_torch_precision(),rhs=rhs,unit=unit,subtract=rhs)
    p=replace(p,calibrateNormalization=True,n_h=1.8,positiveDivergence=True)
    compare(v,lambda backend:functions(backend,p),9e-3 if v[0].dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('sector',[0,1,3])
def test_sector_stencils_and_three_dimensional_xy_policy(device,rhs,explicit,sector):
    q,r,d,p,g,e,v,functions=case(device,3,get_torch_precision(),rhs=rhs,explicit=explicit,subtract=rhs,sector=sector)
    compare(v,functions,9e-3 if v[0].dtype==torch.float32 else 3e-7)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
def test_reference_actual_gradcheck(device,kernel,rhs,explicit):
    if get_torch_precision()!=torch.float64:pytest.skip('Actual finite differences require FP64')
    dimension=1+list(sph.KernelFunctions).index(kernel)%3
    *_,v,functions=case(device,dimension,kernel=kernel,rhs=rhs,explicit=explicit,subtract=rhs)
    assert torch.autograd.gradcheck(functions('warp'),v,fast_mode=True,atol=8e-5,rtol=2e-3,nondet_tol=1e-12 if device=='cuda' else 0.)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('support_value',[0.,-.8])
def test_nonpositive_raw_support_values_and_reverse(device,kernel,dimension,rhs,explicit,support_value):
    q,r,d,p,g,e,v,functions=case(device,dimension,get_torch_precision(),kernel,rhs=rhs,explicit=explicit,subtract=rhs)
    v=list(v)
    for k in ((1,) if explicit else (2,3)):v[k]=torch.full_like(v[k],support_value).requires_grad_()
    compare(tuple(v),functions,2e-2 if v[0].dtype==torch.float32 else 3e-6)


@pytest.mark.parametrize('support',list(sph.SupportScheme))
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
def test_mixed_support_policies(device,support,rhs,explicit):
    q,r,d,p,g,e,v,functions=case(device,dtype=get_torch_precision(),support=support,rhs=rhs,explicit=explicit,subtract=rhs)
    v=list(v)
    if explicit:v[1]=(v[1].detach()*v[1].new_tensor([1.,-1.,1.,-1.])).requires_grad_()
    else:v[2]=-v[2];v[3]=v[3]
    compare(tuple(v),functions,2e-2 if v[0].dtype==torch.float32 else 3e-6)


@pytest.mark.parametrize('direction',list(sph.OperationDirection))
@pytest.mark.parametrize('rhs',[False,True])
@pytest.mark.parametrize('explicit',[False,True])
def test_categorical_interaction_directions(device,direction,rhs,explicit):
    q,r,d,p,g,e,v,functions=case(device,dtype=get_torch_precision(),rhs=rhs,explicit=explicit,subtract=rhs)
    p=replace(p,operationMode=direction)
    compare(v,lambda backend:functions(backend,p),9e-3 if v[0].dtype==torch.float32 else 3e-7)
