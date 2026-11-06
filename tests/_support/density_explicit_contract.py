"""Full explicit density product, including its reverse-tracked direction roles."""
import torch
import warpSPHCore as sph
from .density_contract import case as density_case,oracle as density_equation


def case(*args,**kwargs):
    q,r,d,p,g,v,states,_=density_case(*args,**kwargs)
    v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
    t=tuple((x.detach().cos()*.03).requires_grad_() for x in v[:5]);values=(*v,*t)
    def functions(backend,properties=p,adjacency=g):
        if backend=='native':from warpSPHCore.native.api import warpOperationJVP as compute
        else:from warpSPHCore.coreOperations.wp_densityJVP import computeSPHDensityGeometryJVP as compute
        def run(*values):
            qi,rj,domain=states(values[:10]);qt=sph.ParticleTangentState(values[10],values[12],None,None)
            rt=sph.ParticleTangentState(values[11],values[13],values[14],None)
            if backend=='native':return compute(qi,properties,domain,queryTangentState=qt,referenceTangentState=rt,referenceParticles=rj,adjacency=adjacency)
            return compute(qi,domain,properties.kernel,properties.supportMode,adjacency,qt,rj,rt,properties.operationMode)
        return run
    return q,r,d,p,g,values,states,functions


def oracle(values,domain,properties,graph):
    v=values[:10];t=(*values[10:15],*(torch.zeros_like(x) for x in v[5:]))
    _,dy=torch.autograd.functional.jvp(lambda *x:density_equation(x,domain,properties,graph),v,t,create_graph=True)
    return dy


def boundary_case(device,dimension,dtype,kernel,support,delta,role='joint'):
    from dataclasses import replace
    q,r,d,p,g,_,_,_=case(device,dimension,dtype,kernel=kernel,support=support)
    qx=torch.zeros((2,dimension),device=device,dtype=dtype);qx[:,0]=delta;rx=torch.zeros_like(qx);h=qx.new_full((2,),.8);m=h.new_tensor([.4,.9])
    kinds=torch.zeros(2,device=device,dtype=torch.int32);q=sph.ParticleState(qx,h,m,kinds,h);r=replace(q,positions=rx,supports=h.clone())
    d=replace(d,periodic=torch.zeros(dimension,device=device,dtype=torch.bool))
    g=sph.AdjacencyList(None,torch.arange(2,device=device,dtype=sph.get_torch_index_dtype()),torch.ones(2,device=device,dtype=torch.int32),torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype()),2,2)
    values=[qx,rx,h,h.clone(),m,d.min,d.max,m.clone(),h.clone(),h.clone(),torch.full_like(qx,.03),torch.full_like(rx,-.02),h*.03,h*-.02,m.cos()*.03]
    if role!='joint':
        for slot in range(10,15):
            if slot not in ((12,13) if role=='support' else (14,)):values[slot]=torch.zeros_like(values[slot])
    values=tuple(x.detach().clone().requires_grad_(i not in (5,6)) for i,x in enumerate(values))
    def functions(backend):
        if backend=='native':from warpSPHCore.native.api import warpOperationJVP as compute
        else:from warpSPHCore.coreOperations.wp_densityJVP import computeSPHDensityGeometryJVP as compute
        def run(*v):
            qi=replace(q,positions=v[0],supports=v[2],masses=v[7],densities=v[8]);rj=replace(r,positions=v[1],supports=v[3],masses=v[4],densities=v[9]);domain=replace(d,min=v[5],max=v[6])
            qt=sph.ParticleTangentState(v[10],v[12],None,None);rt=sph.ParticleTangentState(v[11],v[13],v[14],None)
            if backend=='native':return compute(qi,p,domain,queryTangentState=qt,referenceTangentState=rt,referenceParticles=rj,adjacency=g)
            return compute(qi,domain,kernel,support,g,qt,rj,rt,p.operationMode)
        return run
    return values,functions
