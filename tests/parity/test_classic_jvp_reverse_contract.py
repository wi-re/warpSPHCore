"""Reverse ordinary forward-AD tangents, independently of the explicit JVP API."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.classic_jvp_contract import VARIANTS,FIELD_VARIANTS,case


@pytest.fixture(scope='module',autouse=True)
def reference_runtime():wp.init()


def compare(values,functions,reference="warp"):
    tolerance=6e-3 if values[0].dtype==torch.float32 else 2e-7
    inputs=[x for x in values if x.requires_grad]
    for weighted in (False,True):
        actual=functions('native')(*values);expected=functions(reference)(*values)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
        seed=torch.linspace(-.3,.7,actual.numel(),device=actual.device,dtype=actual.dtype).reshape_as(actual) if weighted else torch.ones_like(actual)
        bars=torch.autograd.grad(actual,inputs,seed,allow_unused=True);wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        for k,a,b,x in zip(range(len(inputs)),bars,wanted,inputs):
            torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=tolerance,atol=tolerance,msg=lambda msg:f'classic product role {k}: {msg}')


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('volume',[False,True])
def test_geometry_tangent_scalar_and_weighted_reverse(device,variant,kernel,volume):
    if variant=='density' and volume:pytest.skip('density has no volume correction')
    i=list(sph.KernelFunctions).index(kernel);dimension=2 if variant=='curl' else 1+i%3
    *_,values,states,functions=case(device,variant,dimension,get_torch_precision(),kernel=kernel,correction='volume' if volume else 'plain')
    compare(values,functions)


@pytest.mark.parametrize('variant',FIELD_VARIANTS)
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('side',['query','reference','both'])
def test_packed_field_only_tangent_reverse_keeps_bounds_and_corrections(device,variant,corrected,side,packed):
    if variant=='interpolate' and side=='query':pytest.skip('interpolation has no query field')
    *_,values,states,functions=case(device,variant,dtype=get_torch_precision(),correction='combined' if corrected else 'plain',field_only=True,packed=packed,side=side)
    compare(values,functions,"warp_explicit" if packed else "warp")


@pytest.mark.parametrize('counts',[False,True])
def test_covariance_counts_remain_discrete_with_tangent_reverse(device,counts):
    *_,values,states,functions=case(device,'covariance',dtype=get_torch_precision(),counts=counts)
    compare(values,functions)


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('field_only',[False,True])
def test_exact_zero_bound_directions_keep_supported_reverse(device,variant,field_only):
    if field_only and variant not in FIELD_VARIANTS:pytest.skip('operator has no fields')
    *_,values,states,functions=case(device,variant,dtype=get_torch_precision(),field_only=field_only,zero_bounds=True)
    compare(values,functions)


@pytest.mark.parametrize('variant',FIELD_VARIANTS)
def test_field_only_grad_h_constant_reverse_matches_explicit_reference(device,variant):
    if variant=='interpolate':pytest.skip('interpolation ignores grad-h')
    *_,values,states,functions=case(device,variant,dtype=get_torch_precision(),correction='combined',field_only=True,gradh=True)
    compare(values,functions,'warp_explicit')


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('corrected',[False,True])
@pytest.mark.parametrize('separation',['self','ultratiny','clamped','epsilon','near','cutoff','outside','interior'])
def test_positive_field_product_matches_frozen_primal_mask(device,mode,kernel,corrected,separation):
    from _support.positive_field_product_contract import case as positive_case
    values,product,reference,_=positive_case(device,mode,dtype=get_torch_precision(),corrected=corrected,kernel=kernel,separation=separation)
    tolerance=8e-3 if values[0].dtype==torch.float32 else 7e-7
    inputs=[x for x in values if x.requires_grad]
    for weighted in (False,True):
        actual=product(*values);expected=reference('warp',*values)
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance,equal_nan=True)
        seed=torch.linspace(-.3,.7,actual.numel(),device=device,dtype=actual.dtype).reshape_as(actual) if weighted else torch.ones_like(actual)
        bars=torch.autograd.grad(actual,inputs,seed,allow_unused=True);wanted=torch.autograd.grad(expected,inputs,seed,allow_unused=True)
        budgets={}
        if corrected and mode!=sph.LaplacianScheme.Naive:
            from _support.positive_field_product_contract import boundary_roundoff_budgets
            budgets=boundary_roundoff_budgets(values,kernel,seed,mode)
        for k,a,b,x in zip(range(len(inputs)),bars,wanted,inputs):
            a=torch.zeros_like(x) if a is None else a;b=torch.zeros_like(x) if b is None else b
            if k in budgets:
                torch.testing.assert_close(a.isnan(),b.isnan());torch.testing.assert_close(a.isposinf(),b.isposinf());torch.testing.assert_close(a.isneginf(),b.isneginf())
                finite=a.isfinite()&b.isfinite();bound=budgets[k]
                assert torch.all((a-b).abs()[finite] <= (tolerance+tolerance*b.abs()+bound)[finite]),f'positive field coefficient role {k}'
            else:torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance,equal_nan=True,msg=lambda msg:f'positive field product role {k}: {msg}')
