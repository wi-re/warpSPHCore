"""Shared Warp scalar/vector periodic geometry, in configured precision."""
from typing import Any
import pytest
import torch
import warp as wp
from warpSPHCore import computeDistanceVec, computeDistance, domainData
from warpSPHCore.type_config import scalar_t, get_torch_precision
from warpSPHCore.math.wp_distance import mod_warp, moduloDistanceComponent
from _support.periodic import cases, exact


@wp.kernel
def evaluate(x:wp.array(dtype=Any), y:wp.array(dtype=Any), domain:domainData,
             delta:wp.array(dtype=Any), norm:wp.array(dtype=scalar_t)):
    i=wp.tid()
    delta[i]=computeDistanceVec(x[i],y[i],domain)
    norm[i]=computeDistance(x[i],y[i],domain)


@pytest.mark.parametrize('dim', [1,2,3])
def test_vector_geometry_local_bits_seams_and_images(device, dim):
    wp.init()
    dtype=get_torch_precision()
    x,y,length=cases(dtype,dim,device)
    flags=torch.tensor([True,False,True][:dim],device=device)
    lower=torch.tensor([-.5,-4.,.125][:dim],dtype=dtype,device=device)
    upper=lower+length
    domain=domainData()
    domain.domainMin=wp.from_torch(lower);domain.domainMax=wp.from_torch(upper)
    domain.periodicity=wp.from_torch(flags);domain.dim=dim
    delta=torch.empty_like(x);norm=torch.empty(len(x),dtype=dtype,device=device)
    vec=wp.types.vector(length=dim,dtype=scalar_t)
    wp.launch(evaluate, len(x), inputs=[wp.from_torch(x,dtype=vec),wp.from_torch(y,dtype=vec),domain,
              wp.from_torch(delta,dtype=vec),wp.from_torch(norm)],device=str(device))
    expected=exact(x.cpu(),y.cpu(),flags.cpu(),length.cpu()).to(device)
    torch.testing.assert_close(delta,expected,rtol=0,atol=2*torch.finfo(dtype).eps)
    assert torch.equal(delta[:4],(x-y)[:4])
    assert torch.equal(delta[:,~flags],(x-y)[:,~flags])
    torch.testing.assert_close(norm,expected.norm(dim=-1),rtol=2*torch.finfo(dtype).eps,atol=2*torch.finfo(dtype).eps)


def test_periodic_position_gradients_away_from_ties(device):
    wp.init()
    dtype=get_torch_precision()
    x=torch.tensor([[.25,.1],[.49,-.49]],dtype=dtype,device=device)
    y=torch.tensor([[.26,.11],[-.49,.49]],dtype=dtype,device=device)
    flags=torch.ones(2,dtype=torch.bool,device=device)
    lower=x.new_full((2,),-.5);upper=x.new_full((2,),.5)
    domain=domainData();domain.domainMin=wp.from_torch(lower);domain.domainMax=wp.from_torch(upper)
    domain.periodicity=wp.from_torch(flags);domain.dim=2
    vec=wp.types.vector(length=2,dtype=scalar_t)
    wx=wp.from_torch(x,dtype=vec,requires_grad=True);wy=wp.from_torch(y,dtype=vec,requires_grad=True)
    delta=wp.empty(2,dtype=vec,device=str(device))
    norm=wp.empty(2,dtype=scalar_t,device=str(device),requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(evaluate,2,inputs=[wx,wy,domain,delta,norm],device=str(device))
    tape.backward(grads={norm:wp.ones(2,dtype=scalar_t,device=str(device))})
    d=exact(x.cpu(),y.cpu(),flags.cpu(),(upper-lower).cpu()).to(device)
    expected=d/d.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(wp.to_torch(wx.grad),expected,rtol=3e-6,atol=3e-7)
    torch.testing.assert_close(wp.to_torch(wy.grad),-expected,rtol=3e-6,atol=3e-7)


@wp.kernel
def evaluate_span(x:wp.array(dtype=scalar_t), y:wp.array(dtype=scalar_t),
                  lo:wp.array(dtype=scalar_t), hi:wp.array(dtype=scalar_t),
                  helper:wp.int32, result:wp.array(dtype=scalar_t)):
    i=wp.tid()
    if helper==0:
        result[i]=moduloDistanceComponent(x[i],y[i],True,lo[0],hi[0])
    else:
        result[i]=mod_warp(x[i]-y[i],lo[0],hi[0])


@pytest.mark.parametrize('helper',[0,1])
@pytest.mark.parametrize('local_nan_seed',[False,True])
def test_period_derivatives_follow_fixed_image_and_skip_unused_bounds(device,helper,local_nan_seed):
    dtype=get_torch_precision()
    x=torch.tensor([.25] if local_nan_seed else [.25,.49],dtype=dtype,device=device)
    y=torch.tensor([.26] if local_nan_seed else [.26,-.49],dtype=dtype,device=device)
    lo=x.new_tensor([-.5]);hi=x.new_tensor([.5])
    arrays=[wp.from_torch(v,requires_grad=True) for v in (x,y,lo,hi)]
    result=wp.empty(len(x),dtype=scalar_t,device=str(device),requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(evaluate_span,len(x),inputs=[*arrays,helper,result],device=str(device))
    seed=torch.full_like(x,float('nan')) if local_nan_seed else torch.ones_like(x)
    tape.backward(grads={result:wp.from_torch(seed)})
    for array,expected in zip(arrays[2:],(0.,0.) if local_nan_seed else (1.,-1.)):
        torch.testing.assert_close(wp.to_torch(array.grad),lo.new_tensor([expected]),rtol=0,atol=0)
    if not local_nan_seed:
        torch.testing.assert_close(wp.to_torch(arrays[0].grad),torch.ones_like(x),rtol=0,atol=0)
        torch.testing.assert_close(wp.to_torch(arrays[1].grad),-torch.ones_like(y),rtol=0,atol=0)
