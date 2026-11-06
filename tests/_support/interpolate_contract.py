"""Fixed-topology interpolation fixtures and independent Torch equations."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .density_contract import case as density_case
from .operators import normalization
from .kernels import shape


def case(device, dimension=2, dtype=torch.float64, support=None, kernel=None,
         field_shape=(), corrected=False, graph_types=None):
    q,r,d,p,g,base,states,_=density_case(device,dimension,dtype,support,kernel,graph_types)
    p=replace(p,operation=sph.WarpOperation.Interpolate)
    size=3
    for width in field_shape:size*=width
    field=torch.linspace(-.4,.7,size,device=device,dtype=dtype).reshape(3,*field_shape)
    values=(*base,field.requires_grad_(),q.supports.new_tensor([.9,1.1]).requires_grad_(),
            q.positions.new_full((2,dimension),.07).requires_grad_(),
            r.supports.new_tensor([.31,.47,.63]).requires_grad_())
    def function(backend,properties=p,adjacency=g):
        from warpSPHCore.coreOperations.native.wrapper import sphOperation
        def run(*v):
            qi,rj,domain=states(v[:10])
            options=dict(referenceParticles=rj,adjacency=adjacency,referenceValues=v[10])
            if corrected:options.update(crkState=sph.CRKState(v[11],v[12],None,None),referenceVolumes=v[13])
            return sphOperation(qi,properties,domain,backend=backend,**options)
        return run
    return q,r,d,p,g,values,states,function


def oracle(v,domain,properties,graph,corrected=False):
    qx,rx,qh,rh,m,lo,hi=v[:7]
    dx=qx[:,None]-rx[None];length=hi-lo
    dx=torch.where(domain.periodic,(dx+length/2).remainder(length)-length/2,dx)
    distance=dx.norm(dim=-1)
    def value(h):
        ratio=distance/h;active=ratio<=1
        if properties.kernel==sph.KernelFunctions.AdhesionKernel:
            active=active&(ratio>.5);ratio=torch.where(active,ratio,ratio.new_tensor(.75))
        return torch.where(active,normalization(properties.kernel,domain.dim)*shape(ratio,properties.kernel,domain.dim)/h**domain.dim,0.)
    left,right=qh[:,None],rh[None];support=properties.supportMode.value
    w=(value(left)+value(right))/2 if support in (14,15) else value(left if support==11 else right if support==12 else (left+right)/2 if support==13 else torch.maximum(left,right))
    if corrected:w=w*v[11][:,None]*(1+(v[12][:,None]*dx).sum(-1))
    volume=v[13] if corrected else m/v[9]
    field=v[10].reshape(len(rx),-1)
    result=[]
    for row in range(len(qx)):
        start=int(graph.edgeOffsets[row]);count=int(graph.numNeighbors[row]);ids=graph.paddedJ[start:start+count].long()
        result.append((w[row,ids,None]*volume[ids,None]*field[ids]).sum(0))
    result=torch.stack(result) if result else qx.new_empty((0,field.shape[1]))+sum(x.sum()*0 for x in v)
    return result.reshape(len(qx),*v[10].shape[1:])
