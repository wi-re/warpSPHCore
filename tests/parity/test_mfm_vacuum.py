"""Exact vacuum-opening controls, independent of the HLLC retry oracle."""
import numpy as np
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.mfm import MeshlessWarp
from warpSPHCore.mfm.riemann import robustFaceFlux
from warpSPHCore.mfm.wp_mfm import faceFluxPair
from warpSPHCore.autograd.stream import torchWarpStream

scalar_t=sph.scalar_t
vec_t=wp.types.vector(length=2,dtype=scalar_t)

@wp.kernel(enable_backward=False)
def sample(states:wp.array2d(dtype=scalar_t),out:wp.array2d(dtype=scalar_t),mode:wp.int32):
    i=wp.tid()
    mass,momentum,energy,contact,pressure=faceFluxPair(states[i,0],states[i,1],vec_t(0.,.25),states[i,2],
        states[i,3],states[i,4],vec_t(0.,-.4),states[i,5],scalar_t(1.4),mode,vec_t(1.,0.),vec_t(3.,-1.))
    out[i,0]=mass;out[i,1]=momentum[0];out[i,2]=momentum[1];out[i,3]=energy;out[i,4]=pressure


def inputs_and_exact(device,dtype):
    g=1.4
    data=np.array([[1.,-1.,1e-8,1.,1.,1e-8],[1.,0.,1/g,1.,12.,1/g],
        [1.,-12.,1/g,1.,0.,1/g],[1.,2.,1/g,1.,14.,1/g],[1.,-14.,1/g,1.,-2.,1/g]])
    a=2/(g+1);rho=a**(2/(g-1));pressure=a**(2*g/(g-1))/g
    primitive=[(0.,0.,0.,0.),(rho,a,pressure,.25),(rho,-a,pressure,-.4),(1.,2.,1/g,.25),(1.,-2.,1/g,-.4)]
    expected=[];frame=np.array([3.,-1.])
    for r,u,p,tangent in primitive:
        mass=r*u;momentum=np.array([r*u*u+p,mass*tangent])
        energy=(p/(g-1)+.5*r*(u*u+tangent*tangent)+p)*u
        expected.append([mass,*(momentum+mass*frame),energy+frame@momentum+.5*(frame@frame)*mass])
    return torch.tensor(data,device=device,dtype=dtype),torch.tensor(expected,device=device,dtype=dtype)


@pytest.mark.parametrize('mode',['MFM','MFV'])
def test_physical_vacuum_is_not_retried_as_zero_velocity(device,mode):
    dtype=sph.get_torch_precision();data,exact=inputs_and_exact(device,dtype)
    rL,uL,pL,rR,uR,pR=data.unbind(1);vtL=torch.zeros((len(data),2),device=device,dtype=dtype);vtR=vtL.clone();vtL[:,1]=.25;vtR[:,1]=-.4
    normal=torch.zeros_like(vtL);normal[:,0]=1;frame=torch.full_like(vtL,-1);frame[:,0]=3
    states=(rL,uL,vtL,pL,rR,uR,vtR,pR)
    expected=torch.zeros_like(exact)if mode=='MFM'else exact
    flux,contact,pressure,stage=robustFaceFlux(states,states,1.4,mode,normal,frame,torch.ones_like(rL))
    tolerance=3e-5 if dtype==torch.float32 else 3e-12
    torch.testing.assert_close(flux,expected,rtol=tolerance,atol=tolerance)
    assert (stage==0).all() and (pressure==0).all() and torch.isfinite(contact).all()
    out=torch.empty((len(data),5),device=device,dtype=dtype)
    with torchWarpStream(data.device):
        wp.launch(sample,dim=len(data),inputs=[wp.from_torch(data),wp.from_torch(out),0 if mode=='MFM'else 1],device=wp.device_from_torch(data.device))
    torch.testing.assert_close(out[:,:4],expected,rtol=tolerance,atol=tolerance)
    assert (out[:,4]==0).all()


@pytest.mark.parametrize('owner',['warp','native'])
@pytest.mark.parametrize('mode',['MFM','MFV'])
def test_public_expanding_cold_pair_has_no_artificial_pressure_work(device,owner,mode):
    dtype=sph.get_torch_precision();x=torch.tensor([[-.2],[.2]],device=device,dtype=dtype);h=torch.ones(2,device=device,dtype=dtype)
    domain=sph.DomainDescription(h.new_tensor([-1.]),h.new_tensor([1.]),torch.zeros(1,device=device,dtype=torch.bool),1)
    adjacency=sph.AdjacencyList(None,torch.tensor([0,1,0,1],device=device,dtype=sph.get_torch_index_dtype()),
        torch.full((2,),2,device=device,dtype=torch.int32),torch.tensor([0,2],device=device,dtype=sph.get_torch_offset_dtype()),2,2,numEdges=4)
    geom=MeshlessWarp.build(x,h,domain,sph.KernelFunctions.CubicSpline,adjacency=adjacency,closure='none',backend=owner)
    result=geom.rates(h,h.new_tensor([[-1.],[1.]]),h*1e-8,1.4,mode=mode,order=1,dt=0.,retry=True)
    torch.testing.assert_close(result,torch.zeros_like(result),rtol=0,atol=0)


def test_exact_vacuum_fan_keeps_torch_reference_derivatives(device):
    from warpSPHCore.mfm.riemann import faceFlux
    data,_=inputs_and_exact(device,torch.float64);data=data[1:2]
    arguments=tuple(v.detach().requires_grad_()for v in data.unbind(1))
    vtL=torch.tensor([[0.,.25]],device=device,dtype=torch.float64,requires_grad=True)
    vtR=torch.tensor([[0.,-.4]],device=device,dtype=torch.float64,requires_grad=True)
    normal=torch.tensor([[1.,0.]],device=device,dtype=torch.float64)
    frame=torch.tensor([[3.,-1.]],device=device,dtype=torch.float64)
    def call(rl,ul,pl,rr,ur,pr,tl,tr):
        return faceFlux(rl,ul,tl,pl,rr,ur,tr,pr,1.4,'MFV',normal,frame)[0]
    assert torch.autograd.gradcheck(call,(*arguments,vtL,vtR),eps=1e-6,atol=1e-5)
