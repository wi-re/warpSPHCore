"""Covariance fixtures and an independent fixed-CSR matrix equation."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .laplacian_contract import case as laplacian_case
from .operators import normalization
from .kernels import shape


def case(device,dimension=2,dtype=torch.float64,support=None,kernel=None,corrected=False,graph_types=None,counts=False):
    q,r,d,p,g,v,states,_=laplacian_case(device,dimension,dtype,support,kernel,corrected=corrected,graph_types=graph_types)
    p=replace(p,operation=sph.WarpOperation.Covariance)
    def functions(backend,properties=p,adjacency=g,return_counts=counts):
        from warpSPHCore.coreOperations.native.wrapper import sphOperation
        def run(*v):
            qi,rj,domain=states(v[:10]);options=dict(referenceParticles=rj,adjacency=adjacency,covarianceReturnNumNeighbors=return_counts)
            if corrected:options.update(referenceVolumes=v[13],crkState=sph.CRKState(v[11],v[12],v[15],v[16]),
                gradHState=sph.GradHState(v[19],v[17]),renormalizationState=v[18])
            return sphOperation(qi,properties,domain,v[14],v[10],backend=backend,**options)
        return run
    return q,r,d,p,g,v,states,functions


@torch.enable_grad()
def oracle(v,d,p,g,corrected=False,counts=False):
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
        scale=normalization(p.kernel,dim);eps=1e-7 if qx.dtype==torch.float32 else 1e-15
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
        w=w*A[:,None]*poly
    volume=v[13] if corrected else m/v[9]
    contribution=-dx[:,:,:,None]*volume[None,:,None,None]*gradient[:,:,None,:]
    result=[];number=[]
    for i in range(len(qx)):
        start=int(g.edgeOffsets[i]);count=int(g.numNeighbors[i]);ids=g.paddedJ[start:start+count].long()
        result.append(contribution[i,ids].sum(0));number.append((w[i,ids]>0).sum(dtype=torch.int32))
    matrix=torch.stack(result) if result else qx.new_empty((0,dim,dim))+sum(x.sum()*0 for x in v)
    return (matrix,torch.stack(number) if number else g.numNeighbors.new_empty((0,))) if counts else matrix
