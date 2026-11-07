"""CRK corrected consistency ratio and an independent scalar equation."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .crk_moments_contract import case as moment_case
from .operators import normalization
from .kernels import shape


def case(device, dimension=2, dtype=torch.float64, kernel=None, graph_types=None,
         volume_mode='reference', coefficient_mode='full'):
    q,r,d,p,g,v,states,_ = moment_case(device,dimension,dtype,kernel=kernel,graph_types=graph_types)
    values=(*v, torch.linspace(.9,1.2,len(q.positions),device=device,dtype=dtype), q.positions.cos()*.04,
        torch.zeros_like(q.positions),torch.zeros((len(q.positions),dimension,dimension),device=device,dtype=dtype),
        torch.ones_like(r.supports),r.positions.cos()*.02,torch.zeros_like(r.positions),
        torch.zeros((len(r.positions),dimension,dimension),device=device,dtype=dtype))
    values=tuple(x.detach().requires_grad_() for x in values)
    def functions(backend,props=p,adjacency=g):
        def run(*x):
            qi,rj,domain=states(x[:10])
            coefs=None if coefficient_mode=='none' else sph.CRKState(*x[12:16])
            if coefficient_mode=='tuple':coefs=(coefs,sph.CRKState(*x[16:20]))
            query_volume=x[10] if volume_mode=='query' else x[11] if volume_mode=='reference' else None
            reference_volume=x[10] if volume_mode=='reference' else None
            if backend=='native':from warpSPHCore.crk.native.nt_density import computeCRKDensity as compute
            else:from warpSPHCore.crk.crk_density import _computeCRKDensity_stateBackend as compute
            return compute(qi,props,domain,coefs,query_volume,reference_volume,adjacency,rj)
        return run
    return q,r,d,p,g,values,states,functions


def equation(values,domain,properties,graph,volume_mode='reference',coefficient_mode='full'):
    if properties.operationMode!=sph.OperationDirection.TrueAllToToAll:
        raise ValueError('Independent equation uses unfiltered fixtures')
    dx=values[0][:,None]-values[1][None];length=values[6]-values[5]
    dx=torch.where(domain.periodic,(dx+length/2).remainder(length)-length/2,dx)
    radius=dx.norm(dim=-1);h=values[3][None];ratio=radius/h
    active=(h>0)&(ratio<=1)
    if properties.kernel==sph.KernelFunctions.AdhesionKernel:
        active=active&(ratio>.5);ratio=torch.where(active,ratio,ratio.new_tensor(.75))
    raw=torch.where(active,normalization(properties.kernel,domain.dim)*shape(ratio,properties.kernel,domain.dim)/h**domain.dim,0.)
    corrected=values[12][:,None]*(1+(dx*values[13][:,None]).sum(-1))*raw
    if coefficient_mode=='none':corrected=corrected*0
    volume=values[10] if volume_mode!='none' else torch.zeros_like(values[10])
    rows=[]
    for i in range(len(values[0])):
        start=int(graph.edgeOffsets[i]);count=int(graph.numNeighbors[i]);ids=graph.paddedJ[start:start+count].long()
        numerator=(values[4][ids]*volume[ids]*corrected[i,ids]).sum()
        denominator=(volume[ids]*volume[ids]*corrected[i,ids]).sum()
        rows.append(torch.where(denominator!=0,numerator/torch.where(denominator!=0,denominator,1.),0.))
    return torch.stack(rows)
