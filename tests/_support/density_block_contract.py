"""Fixed-geometry position-block fixtures and independent smooth Hessians."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .density_contract import case as density_case
from .operators import normalization
from .kernels import shape


def case(device,dimension=2,dtype=torch.float64,support=None,kernel=None,graph_types=None,omit_reference=False):
    q,r,d,p,g,values,states,_=density_case(device,dimension,dtype,support,kernel,graph_types)
    q,r,d=states(tuple(v.detach() for v in values))
    mass=r.masses.clone().requires_grad_()
    left=torch.linspace(-.23,.41,q.positions.numel(),device=device,dtype=dtype).reshape_as(q.positions).requires_grad_()
    right=torch.linspace(.19,-.37,r.positions.numel(),device=device,dtype=dtype).reshape_as(r.positions).requires_grad_()
    v=(mass,left) if omit_reference else (mass,left,right)
    def functions(backend,properties=p,adjacency=g,row_start=0):
        if backend=='native':from warpSPHCore.coreOperations.native.nt_density_hvp import computeSPHDensityPositionHVP as compute
        else:from warpSPHCore.coreOperations.wp_densityHVP import computeSPHDensityPositionHVP as compute
        def run(m,a,b=None):
            options={'_rowStart':row_start} if backend=='native' else {}
            return compute(q,d,properties.kernel,properties.supportMode,adjacency,a,
                replace(r,masses=m),b,properties.operationMode,**options)
        return run
    return q,r,d,p,g,v,functions


@torch.enable_grad()
def smooth_matrices(q,r,d,p):
    dim=d.dim;dx=q.positions[:,None]-r.positions[None];length=d.max-d.min
    dx=torch.where(d.periodic,(dx+length/2).remainder(length)-length/2,dx)
    distance=dx.norm(dim=-1);eps=1e-7 if dx.dtype==torch.float32 else 1e-15
    def evaluate(h):
        z=(distance/h).detach().requires_grad_();active=z<=1
        safe=z
        if p.kernel==sph.KernelFunctions.AdhesionKernel:
            active=active&(z>.5);safe=torch.where(active,z,z.new_tensor(.75))
        f=torch.where(active,shape(safe,p.kernel,dim),0.)
        first=torch.autograd.grad(f.sum(),z,create_graph=True)[0]
        second=torch.autograd.grad(first.sum(),z)[0]
        k1=normalization(p.kernel,dim)*first.detach()/h**(dim+1)
        k2=normalization(p.kernel,dim)*second.detach()/h**(dim+2)
        outer=dx[:,:,:,None]*dx[:,:,None,:]
        identity=torch.eye(dim,device=dx.device,dtype=dx.dtype)
        a=outer/(distance.square()+eps**2*h**2)[:,:,None,None]
        a=torch.where(((distance/h)<eps)[:,:,None,None]&identity.bool(),1.,a)
        b=-outer/(distance**3+eps**3*h**3)[:,:,None,None]+identity/(distance+eps**2*h)[:,:,None,None]
        return (a*k2[:,:,None,None]+b*k1[:,:,None,None]).detach()
    hi=q.supports[:,None];hj=r.supports[None];support=p.supportMode.value
    if support==15:return (evaluate(hi)+evaluate(hj))/2
    return evaluate(hi if support==11 else hj if support==12 else (hi+hj)/2 if support==13 else torch.maximum(hi,hj))


def oracle(v,q,r,d,p,g,H=None,row_start=0):
    H=smooth_matrices(q,r,d,p) if H is None else H
    mass,left=v[:2];right=v[2] if len(v)>2 else None;rows=[]
    for i in range(len(q.positions)):
        begin=int(g.edgeOffsets[i]);count=int(g.numNeighbors[i]);ids=g.paddedJ[begin:begin+count].long()
        ids=ids[ids!=i+row_start]
        vectors=left[i]-(right[ids] if right is not None else torch.zeros_like(r.positions[ids]))
        rows.append((mass[ids,None]*torch.bmm(H[i,ids],vectors.unsqueeze(-1)).squeeze(-1)).sum(0))
    return torch.stack(rows) if rows else left.new_empty(left.shape)+sum(x.reshape(-1)[:0].sum() for x in v)
