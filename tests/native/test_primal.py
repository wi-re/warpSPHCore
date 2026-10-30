"""Expanded native primals against a dense, independently differentiated oracle."""
from dataclasses import replace
import math
import os
import pytest
import torch
import warpSPHCore as sph
if sph.backend != 'native': pytest.skip('native process only', allow_module_level=True)
from test_kernels import shape, SPECS, device


def fixture(device, dim, rank):
    gen=torch.Generator().manual_seed(771)
    def rand(*size): return torch.rand(size,generator=gen,dtype=torch.float64).to(device)
    def state(n):
        return sph.ParticleState((rand(n,dim)-.5)*1.6,rand(n)*.5+.6,rand(n)+.2,
                                 (torch.arange(n,device=device)%4).int(),rand(n)+.7)
    q,r=state(7),state(5)
    domain=sph.DomainDescription(q.positions.new_full((dim,),-1),q.positions.new_full((dim,),1),
                                torch.tensor([True]+[False]*(dim-1),device=device),dim)
    return q,r,domain,rand(7,*([dim]*rank))-.5,rand(5,*([dim]*rank))-.5


def normalization(kernel, dim):
    if kernel.name in SPECS:
        spec=SPECS[kernel.name]['C'];return eval(spec.get(dim,spec.get(str(dim))),{'__builtins__':{},'pi':math.pi})
    return {'Poly6':[35/32,4/math.pi,315/(64*math.pi)],'Spiky':[2,10/math.pi,15/math.pi],
            'ViscosityKernel':[15/(2*math.pi),10/(9*math.pi),15/(2*math.pi)],
            'CohesionKernel':[32/math.pi]*3,'AdhesionKernel':[.007]*3}[kernel.name][dim-1]


def dense(q,r,domain,prop,qf,rf, *, differentiable=False):
    D=domain.dim;n,m=len(q.positions),len(r.positions)
    length=domain.max-domain.min
    dx=q.positions[:,None]-r.positions[None]
    dx=torch.where(domain.periodic,torch.remainder(dx+length/2,length)-length/2,dx)
    radius=dx.norm(dim=-1); hi=q.supports[:,None];hj=r.supports[None]
    h={11:hi,12:hj,13:(hi+hj)/2,14:torch.maximum(hi,hj),15:torch.maximum(hi,hj)}[prop.supportMode.value]
    def kernel(h):
        z=(radius/h) if differentiable else (radius/h).detach()
        if not z.requires_grad:z=z.requires_grad_()
        active=(z<=1)
        if prop.kernel==sph.KernelFunctions.AdhesionKernel:
            active &= z>.5
            safe=torch.where(active,z,z.new_tensor(.75))
        else:safe=z
        f=torch.where(active,shape(safe,prop.kernel,D),0)
        d1=torch.autograd.grad(f.sum(),z,create_graph=True)[0]
        d2=torch.autograd.grad(d1.sum(),z,create_graph=differentiable,retain_graph=True)[0]
        if not differentiable:f,d1,d2=f.detach(),d1.detach(),d2.detach()
        C=normalization(prop.kernel,D)
        w=C*f/h**D
        g=(C*d1/h**(D+1))[...,None]*dx/radius.clamp_min(1e-15)[...,None]
        re=radius+1e-15*h
        lap=radius.square()/re.square()*C*d2/h**(D+2)+(-radius.square()/re**3+D/re)*C*d1/h**(D+1)
        return w,g,torch.where((radius/h>=1e-15)&active,lap,0)
    w,g,lap=kernel(h)
    if prop.supportMode.value in (14,15):
        wi,gi,li=kernel(hi);wj,gj,lj=kernel(hj);w=(wi+wj)/2;g=(gi+gj)/2
        if prop.supportMode.value==15:lap=(li+lj)/2
    active=radius<h
    if prop.operationMode==sph.OperationDirection.AllToAll:active &= (r.kinds!=2)[None]
    elif prop.operationMode==sph.OperationDirection.FluidToBoundary:active &= (q.kinds==1)[:,None]&(r.kinds==0)[None]
    elif prop.operationMode!=sph.OperationDirection.TrueAllToToAll:raise AssertionError('oracle direction not defined')
    w=w*active;g=g*active[...,None];lap=lap*active
    volume=r.masses/r.densities
    if prop.operation==sph.WarpOperation.Density:return (w*r.masses).sum(1)
    if prop.operation==sph.WarpOperation.Covariance:return -torch.einsum('ij,ija,ijb->iab',active*volume,dx,g)
    field_shape=qf.shape[1:]; F=math.prod(field_shape)
    fi=qf.reshape(n,F)[:,None];fj=rf.reshape(m,F)[None]
    if prop.operation==sph.WarpOperation.Interpolate:return torch.einsum('ij,jc->ic',w*volume,rf.reshape(m,F)).reshape(n,*field_shape)
    if prop.operation==sph.WarpOperation.Laplacian:
        weighted=(fj-fi)*(r.masses*q.densities[:,None]/r.densities.square())[...,None] if prop.gradientMode==sph.GradientScheme.Symmetric else (fj-fi)*volume[:,None]
    elif prop.gradientMode==sph.GradientScheme.Naive:weighted=fj*volume[:,None]
    elif prop.gradientMode==sph.GradientScheme.Symmetric:
        weighted=r.masses[None,:,None]*q.densities[:,None,None]*(fi/q.densities[:,None,None].square()+fj/r.densities[None,:,None].square())
    else:weighted=(fj-fi if prop.gradientMode==sph.GradientScheme.Difference else fj+fi)*volume[:,None]
    weighted=weighted.expand(n,m,F)
    if prop.operation==sph.WarpOperation.Gradient:return torch.einsum('ijc,ijd->icd',weighted,g).reshape(n,*field_shape,D)
    if prop.operation==sph.WarpOperation.Divergence:
        if prop.divergenceDotMode:out=torch.einsum('ijdc,ijd->ic',weighted.reshape(n,m,D,F//D),g)
        else:out=torch.einsum('ijcd,ijd->ic',weighted.reshape(n,m,F//D,D),g)
        return out.reshape(n,*field_shape[:-1])
    if prop.operation==sph.WarpOperation.Curl:
        if D==1:return torch.zeros(n,device=qf.device,dtype=qf.dtype)
        values=weighted.reshape(n,m,D,F//D)
        if D==2:return (g[:,:,0,None]*values[:,:,1]-g[:,:,1,None]*values[:,:,0]).sum(1).reshape(n,*(field_shape[1:] or (1,)))
        return torch.linalg.cross(g[...,None].expand_as(values),values,dim=2).sum(1).reshape(n,*field_shape)
    re=radius+1e-8*h
    if prop.laplacianMode==sph.LaplacianScheme.Naive:out=(weighted*lap[...,None]).sum(1)
    elif prop.laplacianMode==sph.LaplacianScheme.Dot:
        normals=dx/re[...,None];F_ab=(normals*g).sum(-1)/re
        blocks=weighted.reshape(n,m,F//D,D)
        contribution=(blocks-(D+2)*(blocks*normals[:,:,None]).sum(-1)[...,None]*normals[:,:,None])*F_ab[:,:,None,None]
        out=contribution.sum(1).reshape(n,F)
    else:
        denominator=re*(radius+1e-12*h) if prop.laplacianMode==sph.LaplacianScheme.Default else re.square()
        factor=-2*(g*dx).sum(-1)/denominator
        out=(weighted*factor[...,None]).sum(1)
    return out.reshape(n,*field_shape)


OPS=[sph.WarpOperation.Interpolate,sph.WarpOperation.Gradient,sph.WarpOperation.Divergence,
     sph.WarpOperation.Curl,sph.WarpOperation.Laplacian,sph.WarpOperation.Covariance]

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
def test_primal_fields_and_schemes(device,op,dim,rank,scheme):
    if op==sph.WarpOperation.Divergence and rank==0:pytest.skip('requires vector field')
    if op==sph.WarpOperation.Curl and rank==0 and dim>1:pytest.skip('requires vector field')
    q,r,d,qf,rf=fixture(device,dim,rank)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,gradientMode=scheme,
        supportMode=sph.SupportScheme.KernelMeanSymmetric,operationMode=sph.OperationDirection.AllToAll)
    actual=sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r)
    torch.testing.assert_close(actual,dense(q,r,d,prop,qf,rf),rtol=2e-10,atol=2e-10)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('support',list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('op',[sph.WarpOperation.Density,sph.WarpOperation.Gradient,sph.WarpOperation.Interpolate])
def test_kernel_family_in_primals(device,kernel,support,op):
    q,r,d,qf,rf=fixture(device,2,0)
    prop=sph.OperationProperties(kernel,operation=op,gradientMode=sph.GradientScheme.Difference,
        supportMode=support,operationMode=sph.OperationDirection.TrueAllToToAll)
    actual=sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r)
    torch.testing.assert_close(actual,dense(q,r,d,prop,qf,rf),rtol=3e-9,atol=3e-9)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('rank',[0,1,2])
def test_laplacian_variants(device,mode,dim,scheme,rank):
    if rank==0 and mode==sph.LaplacianScheme.Dot and dim>1:pytest.skip('requires vector field')
    q,r,d,qf,rf=fixture(device,dim,rank)
    prop=sph.OperationProperties(sph.KernelFunctions.CubicSpline,operation=sph.WarpOperation.Laplacian,
        gradientMode=scheme,laplacianMode=mode,supportMode=sph.SupportScheme.SuperSymmetric,
        operationMode=sph.OperationDirection.FluidToBoundary)
    torch.testing.assert_close(sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r),dense(q,r,d,prop,qf,rf),rtol=2e-10,atol=2e-10)


def test_constant_laplacian_and_tensor_divergence_axes(device):
    q,r,d,qf,rf=fixture(device,3,2)
    for scheme in sph.GradientScheme:
        for mode in sph.LaplacianScheme:
            prop=sph.OperationProperties(sph.KernelFunctions.Wendland6,operation=sph.WarpOperation.Laplacian,
                gradientMode=scheme,laplacianMode=mode)
            out=sph.warpOperation(q,prop,d,torch.ones_like(qf),torch.ones_like(rf),referenceParticles=r)
            assert out.count_nonzero()==0
    for dot in (False,True):
        prop=sph.OperationProperties(sph.KernelFunctions.Wendland6,operation=sph.WarpOperation.Divergence,
            gradientMode=sph.GradientScheme.Difference,divergenceDotMode=dot)
        torch.testing.assert_close(sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r),dense(q,r,d,prop,qf,rf),rtol=2e-10,atol=2e-10)
