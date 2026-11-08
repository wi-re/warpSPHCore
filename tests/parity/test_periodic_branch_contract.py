"""Unwrapped displacement has no box-bound AD dependency."""
from importlib import import_module
import pytest
import torch
import warpSPHCore as sph
import warp as wp


@pytest.fixture(scope='module',autouse=True)
def initialize_reference():wp.init()


def case(device,variant,wrapped=False):
    module=import_module('_support.'+variant+'_contract')
    data=module.case(device,2,sph.get_torch_precision())
    if variant=='polyfit':q,r,d,p,a,e,v,functions=data
    else:q,r,d,p,a,v,states,functions=data
    with torch.no_grad():
        d.periodic.fill_(True);v[5].zero_();v[6].fill_(1)
        v[0].fill_(.2);v[1].fill_(.1);v[2].fill_(1);v[3].fill_(1)
        a.numNeighbors.zero_();a.numNeighbors[0]=1;a.paddedJ[0]=1
        if wrapped:v[0][0].add_(2)
    def run(values,backend='native'):return functions(backend)(*values)
    return v,run


def close(a,b):
    tol=4e-5 if a.dtype==torch.float32 else 5e-10
    torch.testing.assert_close(a,b,rtol=tol,atol=tol,equal_nan=True)


@pytest.mark.parametrize('variant',['density','gradient','polyfit'])
def test_unwrapped_nan_seed_does_not_reach_bounds(device,variant):
    v,run=case(device,variant)
    for backend in ('native','warp'):
        y=run(v,backend)
        g=torch.autograd.grad(y,(v[5],v[6]),torch.full_like(y,float('nan')),allow_unused=True)
        for k,b in zip((5,6),g):assert b is None or torch.equal(b,torch.zeros_like(v[k]))


@pytest.mark.parametrize('variant',['density','gradient','polyfit'])
def test_unwrapped_nan_bounds_do_not_pollute_forward(device,variant):
    v,run=case(device,variant)
    with torch.autograd.forward_ad.dual_level():
        dual=tuple(torch.autograd.forward_ad.make_dual(x,torch.full_like(x,float('nan')) if k in (5,6) else torch.zeros_like(x)) for k,x in enumerate(v))
        tangent=torch.autograd.forward_ad.unpack_dual(run(dual)).tangent
    assert torch.isfinite(tangent).all()
    assert torch.equal(tangent,torch.zeros_like(tangent))


@pytest.mark.parametrize('variant',['density','gradient','polyfit'])
def test_wrapped_bounds_products_match_reference(device,variant):
    v,run=case(device,variant,wrapped=True)
    a,b=run(v),run(v,'warp');close(a,b)
    seed=torch.linspace(.2,.8,a.numel(),device=device,dtype=a.dtype).reshape_as(a)
    ga=torch.autograd.grad(a,(v[5],v[6]),seed)
    gb=torch.autograd.grad(b,(v[5],v[6]),seed)
    for x,y in zip(ga,gb):close(x,y)
    t=tuple(torch.full_like(x,.003*(k+1)) if k in (5,6) else torch.zeros_like(x) for k,x in enumerate(v))
    with torch.autograd.forward_ad.dual_level():
        dual=tuple(torch.autograd.forward_ad.make_dual(x,d) for x,d in zip(v,t))
        product=torch.autograd.forward_ad.unpack_dual(run(dual)).tangent
    close((product*seed).sum(),sum((g*t[k]).sum() for k,g in zip((5,6),ga)))
