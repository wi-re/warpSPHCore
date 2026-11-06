"""Divergence fixtures and independent fixed-CSR product equations."""
from dataclasses import replace
import math
import torch
import warpSPHCore as sph
from .interpolate_contract import case as interpolate_case
from .operators import normalization
from .kernels import shape


def case(device,dimension=2,dtype=torch.float64,support=None,kernel=None,field_shape=None,corrected=False,graph_types=None,scheme=None,dot=False,consistent=False):
    if field_shape is None:field_shape=(dimension,)
    q,r,d,p,g,v,states,_=interpolate_case(device,dimension,dtype,support,kernel,field_shape,corrected,graph_types)
    p=replace(p,operation=sph.WarpOperation.Divergence,gradientMode=scheme or sph.GradientScheme.Difference,divergenceDotMode=dot)
    field=q.positions.new_tensor([.2,-.3])[:,None].repeat(1,math.prod(field_shape)).reshape(2,*field_shape)
    additions=(field,q.positions.new_full((2,dimension),.03),q.positions.new_full((2,dimension,dimension),-.02),
               r.supports.new_tensor([.92,1.07,.96]),torch.eye(dimension,device=device,dtype=dtype).repeat(2,1,1)*1.03,
               q.supports.new_tensor([.91,1.08]))
    values=(*v,*(x.detach().requires_grad_() for x in additions))
    def functions(backend,properties=p,adjacency=g):
        from warpSPHCore.coreOperations.native.wrapper import sphOperation
        def run(*v):
            qi,rj,domain=states(v[:10])
            options=dict(referenceParticles=rj,adjacency=adjacency,consistentDivergence=consistent)
            if corrected:options.update(referenceVolumes=v[13],crkState=sph.CRKState(v[11],v[12],v[15],v[16]),
                gradHState=sph.GradHState(v[19],v[17]),renormalizationState=v[18])
            return sphOperation(qi,properties,domain,v[14],v[10],backend=backend,**options)
        return run
    return q,r,d,p,g,values,states,functions


@torch.enable_grad()
def oracle(v,d,p,g,corrected=False,consistent=False):
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
        f=torch.where(active,shape(safe,p.kernel,dim),0.)
        first=torch.autograd.grad(f.sum(),z,create_graph=True)[0]
        scale=normalization(p.kernel,dim)
        eps=1e-7 if qx.dtype==torch.float32 else 1e-15
        return scale*f/h**dim,dx/distance.clamp_min(eps)[...,None]*(scale*first/h**(dim+1))[...,None]
    left,right=qh[:,None],rh[None];support=p.supportMode.value
    if support in (14,15):
        wi,gi=kernel(left);wj,gj=kernel(right);w=(wi+wj)/2;gradient=(gi+gj)/2
    else:w,gradient=kernel(left if support==11 else right if support==12 else (left+right)/2 if support==13 else torch.maximum(left,right))
    if corrected:
        A,B,ga,gb=v[11],v[12],v[15],v[16];poly=1+(B[:,None]*dx).sum(-1)
        gradient=A[:,None,None]*B[:,None]*w[...,None]+(A[:,None]*poly)[...,None]*gradient+\
            (poly*w)[...,None]*ga[:,None]+(A[:,None]*w)[...,None]*torch.einsum('idc,ijc->ijd',gb,dx)
        gradient=torch.einsum('idc,ijc->ijd',v[18],gradient)
    volume=v[13] if corrected else m/v[9]
    if consistent:volume=v[13][None,:]*v[9][None,:]/v[8][:,None] if corrected else m[None,:]/v[8][:,None]
    fi=v[14].reshape(len(qx),math.prod(v[14].shape[1:]));fj=v[10].reshape(len(rx),math.prod(v[10].shape[1:]))
    if corrected:
        fj=fj/v[17][:,None];fi=fi/v[19][:,None]
    if p.gradientMode==sph.GradientScheme.Symmetric:
        weighted=m[None,:,None]*(fi[:,None]/v[8][:,None,None]+v[8][:,None,None]*fj[None]/v[9][None,:,None]**2)
    else:weighted=(fj[None] if p.gradientMode==sph.GradientScheme.Naive else fj[None]-fi[:,None] if p.gradientMode==sph.GradientScheme.Difference else fj[None]+fi[:,None])*(volume[:,:,None] if consistent else volume[None,:,None])
    weighted=weighted.expand(len(qx),len(rx),fi.shape[1]);result=[]
    for i in range(len(qx)):
        start=int(g.edgeOffsets[i]);count=int(g.numNeighbors[i]);ids=g.paddedJ[start:start+count].long()
        result.append(torch.einsum('jf,jd->fd',weighted[i,ids],gradient[i,ids]))
    output=torch.stack(result) if result else qx.new_empty((0,fi.shape[1],dim))+sum(x.sum()*0 for x in v)
    field_shape=tuple(v[14].shape[1:])
    if not field_shape:return output.reshape(len(qx))
    axis=0 if p.divergenceDotMode else len(field_shape)-1
    shaped=output.reshape(len(qx),*field_shape,dim)
    return shaped.diagonal(dim1=axis+1,dim2=shaped.dim()-1).sum(-1)
