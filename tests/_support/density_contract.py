"""Small, fixed-topology density contracts shared by native and parity owners."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .operators import normalization
from .kernels import shape


def case(device, dimension=2, dtype=torch.float64, support=None, kernel=None, graph_types=None):
    qx=torch.tensor([[-.18,.12,.03],[.11,-.09,.16]],device=device,dtype=dtype)[:,:dimension]
    rx=torch.tensor([[.04,-.06,.08],[-.37,.17,-.12],[.41,.23,-.15]],device=device,dtype=dtype)[:,:dimension]
    qx=qx.clone();rx=rx.clone();qx[:,0]+=4;rx[:,0]-=2
    q=sph.ParticleState(qx,qx.new_tensor([.84,.97]),qx.new_tensor([.7,.8]),
                        torch.tensor([0,1],device=device,dtype=torch.int32),qx.new_tensor([1.2,1.3]))
    r=sph.ParticleState(rx,rx.new_tensor([1.04,.91,1.11]),rx.new_tensor([.4,.6,.9]),
                        torch.tensor([0,1,2],device=device,dtype=torch.int32),rx.new_tensor([1.1,1.3,1.4]))
    d=sph.DomainDescription(qx.new_full((dimension,),-1),qx.new_full((dimension,),1),
                            torch.tensor([True]+[False]*(dimension-1),device=device),dimension)
    index,offset=graph_types or (sph.get_torch_index_dtype(),sph.get_torch_offset_dtype())
    graph=sph.AdjacencyList(None,torch.arange(3,device=device,dtype=index).repeat(2),
        torch.full((2,),3,device=device,dtype=torch.int32),torch.tensor([0,3],device=device,dtype=offset),2,3)
    p=sph.OperationProperties(kernel or sph.KernelFunctions.Wendland2,operation=sph.WarpOperation.Density,
        supportMode=support or sph.SupportScheme.Gather,operationMode=sph.OperationDirection.TrueAllToToAll)
    values=tuple(v.detach().clone().requires_grad_() for v in
                 (q.positions,r.positions,q.supports,r.supports,r.masses,d.min,d.max,q.masses,q.densities,r.densities))
    def states(v):
        return (replace(q,positions=v[0],supports=v[2],masses=v[7],densities=v[8]),
                replace(r,positions=v[1],supports=v[3],masses=v[4],densities=v[9]),replace(d,min=v[5],max=v[6]))
    def function(backend,properties=p,adjacency=graph):
        from warpSPHCore.coreOperations.native.wrapper import computeDensity
        def run(*v):
            qi,rj,domain=states(v)
            return computeDensity(qi,rj,domain,properties,adjacency,backend=backend)
        return run
    return q,r,d,p,graph,values,states,function


def oracle(values,domain,properties,graph):
    """Independent Torch density expression; active pair topology stays fixed."""
    qx,rx,qh,rh,m,lo,hi=values[:7]
    dx=qx[:,None]-rx[None]
    length=hi-lo
    dx=torch.where(domain.periodic,(dx+length/2).remainder(length)-length/2,dx)
    distance=dx.norm(dim=-1)
    def value(h):
        ratio=distance/h
        active=ratio<=1
        if properties.kernel==sph.KernelFunctions.AdhesionKernel:
            active=active&(ratio>.5)
            ratio=torch.where(active,ratio,ratio.new_tensor(.75))
        return torch.where(active,normalization(properties.kernel,domain.dim)*
                           shape(ratio,properties.kernel,domain.dim)/h**domain.dim,0.)
    left,right=qh[:,None],rh[None]
    support=properties.supportMode.value
    w=(value(left)+value(right))/2 if support in (14,15) else value(
        left if support==11 else right if support==12 else (left+right)/2 if support==13 else torch.maximum(left,right))
    result=[]
    for row in range(len(qx)):
        start=int(graph.edgeOffsets[row]);count=int(graph.numNeighbors[row])
        ids=graph.paddedJ[start:start+count].long()
        result.append((w[row,ids]*m[ids]).sum())
    return torch.stack(result) if result else qx.new_empty(0)+sum(v.sum()*0 for v in values[:7])
