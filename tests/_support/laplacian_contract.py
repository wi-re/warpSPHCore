"""Laplacian fixtures and independent fixed-CSR product equations."""
from dataclasses import replace
import math
import torch
import warpSPHCore as sph
from .interpolate_contract import case as interpolate_case
from .operators import normalization
from .kernels import shape as kernel_shape


def case(device,dimension=2,dtype=torch.float64,support=None,kernel=None,field_shape=(),corrected=False,graph_types=None,scheme=None,mode=None,positive=False):
    q,r,d,p,g,v,states,_=interpolate_case(device,dimension,dtype,support,kernel,field_shape,corrected,graph_types)
    p=replace(p,operation=sph.WarpOperation.Laplacian,gradientMode=scheme or sph.GradientScheme.Difference,
              laplacianMode=mode or sph.LaplacianScheme.Brookshaw,positiveDivergence=positive)
    field=q.positions.new_tensor([.2,-.3])[:,None].repeat(1,math.prod(field_shape)).reshape(2,*field_shape)
    additions=(field,q.positions.new_full((2,dimension),.03),q.positions.new_full((2,dimension,dimension),-.02),
               r.supports.new_tensor([.92,1.07,.96]),torch.eye(dimension,device=device,dtype=dtype).repeat(2,1,1)*1.03,
               q.supports.new_tensor([.91,1.08]))
    values=(*v,*(x.detach().requires_grad_() for x in additions))
    def functions(backend,properties=p,adjacency=g):
        from warpSPHCore.coreOperations.native.wrapper import sphOperation
        def run(*v):
            qi,rj,domain=states(v[:10])
            options=dict(referenceParticles=rj,adjacency=adjacency)
            if corrected:options.update(referenceVolumes=v[13],crkState=sph.CRKState(v[11],v[12],v[15],v[16]),
                gradHState=sph.GradHState(v[19],v[17]),renormalizationState=v[18])
            return sphOperation(qi,properties,domain,v[14],v[10],backend=backend,**options)
        return run
    return q,r,d,p,g,values,states,functions


@torch.enable_grad()
def oracle(v,d,p,g,corrected=False):
    qx,rx,qh,rh,m,lo,hi=v[:7];dim=d.dim
    dx=qx[:,None]-rx[None];length=hi-lo
    dx=torch.where(d.periodic,(dx+length/2).remainder(length)-length/2,dx)
    distance=dx.norm(dim=-1)
    def kernel(h):
        z=distance/h
        if not z.requires_grad:z=z.requires_grad_()
        active=z<=1
        if p.kernel==sph.KernelFunctions.AdhesionKernel:
            active=active&(z>.5);safe=torch.where(active,z,z.new_tensor(.75))
        else:safe=z
        f=torch.where(active,kernel_shape(safe,p.kernel,dim),0.)
        first=torch.autograd.grad(f.sum(),z,create_graph=True)[0]
        second=torch.autograd.grad(first.sum(),z,create_graph=True)[0]
        scale=normalization(p.kernel,dim)
        eps=1e-7 if qx.dtype==torch.float32 else 1e-15
        re=distance+eps*h
        lap=distance**2/re**2*(scale*second/h**(dim+2))+(-distance**2/re**3+dim/re)*(scale*first/h**(dim+1))
        lap=torch.where((z>=eps)&(z<=1),lap,0.)
        return scale*f/h**dim,dx/distance.clamp_min(eps)[...,None]*(scale*first/h**(dim+1))[...,None],lap
    left,right=qh[:,None],rh[None];support=p.supportMode.value
    if support in (14,15):
        wi,gi,li=kernel(left);wj,gj,lj=kernel(right);w=(wi+wj)/2;gradient=(gi+gj)/2
        lap=(li+lj)/2 if support==15 else kernel(torch.maximum(left,right))[2]
    else:w,gradient,lap=kernel(left if support==11 else right if support==12 else (left+right)/2 if support==13 else torch.maximum(left,right))
    if corrected and p.laplacianMode!=sph.LaplacianScheme.Naive:
        A,B,ga,gb=v[11],v[12],v[15],v[16];poly=1+(B[:,None]*dx).sum(-1)
        gradient=A[:,None,None]*B[:,None]*w[...,None]+(A[:,None]*poly)[...,None]*gradient+\
            (poly*w)[...,None]*ga[:,None]+(A[:,None]*w)[...,None]*torch.einsum('idc,ijc->ijd',gb,dx)
        gradient=torch.einsum('idc,ijc->ijd',v[18],gradient)
    volume=v[13] if corrected else m/v[9]
    field_shape=tuple(v[14].shape[1:]);fields=math.prod(field_shape)
    fi=v[14].reshape(len(qx),fields);fj=v[10].reshape(len(rx),fields)
    if corrected:fi=fi/v[19][:,None];fj=fj/v[17][:,None]
    weight=m[None,:]*v[8][:,None]/v[9][None,:]**2 if p.gradientMode==sph.GradientScheme.Symmetric else volume[None,:]
    delta=(fj[None]-fi[:,None])*weight[:,:,None]
    h=left if support==11 else right if support==12 else (left+right)/2 if support==13 else torch.maximum(left,right)
    regularized=distance+1e-8*h;normal=dx/regularized[...,None]
    projection=(gradient*normal).sum(-1)
    if p.laplacianMode==sph.LaplacianScheme.Naive:contribution=delta*lap[...,None]
    elif p.laplacianMode==sph.LaplacianScheme.Dot:
        blocks=delta.reshape(len(qx),len(rx),fields//dim,dim)
        along=(blocks*normal[:,:,None,:]).sum(-1)
        contribution=((blocks-(dim+2)*along[...,None]*normal[:,:,None,:])*(projection/regularized)[:,:,None,None]).reshape(len(qx),len(rx),fields)
    else:
        denominator=distance+(1e-12 if p.laplacianMode==sph.LaplacianScheme.Default else 1e-8)*h
        contribution=-2*delta*(projection/denominator)[...,None]
    if p.laplacianMode!=sph.LaplacianScheme.Naive:contribution=torch.where((distance>0)[...,None],contribution,0.)
    if p.positiveDivergence:
        allowed=(dx*delta[:,:,:dim]).sum(-1)>=0
        mask=torch.arange(fields,device=qx.device)<dim
        contribution=torch.where(allowed[...,None]&mask[None,None,:],contribution,0.)
    result=[]
    for i in range(len(qx)):
        start=int(g.edgeOffsets[i]);count=int(g.numNeighbors[i]);ids=g.paddedJ[start:start+count].long()
        result.append(contribution[i,ids].sum(0))
    output=torch.stack(result) if result else qx.new_empty((0,fields))+sum(x.sum()*0 for x in v)
    return output.reshape(len(qx),*field_shape)
