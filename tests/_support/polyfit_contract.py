"""Fixed-topology polynomial assembly and independent product equations."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .density_contract import case as geometry_case
from .operators import normalization
from .kernels import shape


def case(device,dimension=2,dtype=torch.float64,kernel=None,support=None,*,rhs=False,explicit=False,
         subtract=False,unit=False,sector=-1,order=2,volume=False,graph_types=None):
    q,r,domain,props,graph,base,states,_=geometry_case(device,dimension,dtype,kernel=kernel,support=support,graph_types=graph_types)
    if explicit:
        x=torch.tensor([[-.62,.05,-.04],[-.04,-.35,.1],[.12,.22,-.2],[.68,-.08,.17]],device=device,dtype=dtype)[:,:dimension].clone()
        x[:,0]+=4
        h=x.new_tensor([1.1,1.21,.96,1.08]);m=x.new_tensor([.4,.6,.9,.7]);rho=x.new_tensor([1.1,1.3,1.4,1.2])
        q=sph.ParticleState(x,h,m,torch.arange(4,device=device,dtype=torch.int32)%3,rho);r=q
        ids=torch.arange(4,device=device);neighbors=torch.cat([ids[ids!=i] for i in range(4)]).to(sph.get_torch_index_dtype())
        graph=sph.AdjacencyList(None,neighbors,torch.full((4,),3,device=device,dtype=torch.int32),
            torch.arange(4,device=device,dtype=sph.get_torch_offset_dtype())*3,4,4)
        base=(x,x,h,h,m,domain.min.detach(),domain.max.detach(),m,rho,rho)
    exponents=torch.tensor(sph.monomialExponents(dimension,order),device=device,dtype=torch.int32)
    field=torch.arange(len(r.positions)*2,device=device,dtype=dtype).reshape(len(r.positions),2).cos()*.7
    vol=torch.linspace(.2,.7,len(r.positions),device=device,dtype=dtype)
    values=tuple(x.detach().clone().requires_grad_() for x in (*base,vol,field))
    if explicit:
        # The public helper aliases both physical roles and their directions.
        values=(values[0],values[2],values[4],values[9],values[11],
                *((x.detach().cos()*.03).requires_grad_() for x in (values[0],values[2],values[4],values[9],values[11])))
    def functions(backend,properties=props,adjacency=graph):
        def run(*v):
            if explicit:
                particle=replace(q,positions=v[0],supports=v[1],masses=v[2],densities=v[3])
                tangent=sph.ParticleTangentState(v[5],v[6],v[7],v[8])
                if backend=='native':from warpSPHCore.polyfit.native import nt_polyfit as implementation
                else:from warpSPHCore.polyfit import wp_polyfitJVP as implementation
                options=dict(supportMode=properties.supportMode,operationMode=properties.operationMode,sector=sector,numSectors=4,unitWeights=unit)
                if rhs:return implementation.computePolyRHSGeometryJVP(particle,domain,properties.kernel,adjacency,tangent,exponents,v[4],v[9],subtract,**options)
                return implementation.computePolyMomentsGeometryJVP(particle,domain,properties.kernel,adjacency,tangent,exponents,**options)
            qi,rj,d=states(v[:10])
            if backend=='native':from warpSPHCore.polyfit.native import nt_polyfit as implementation
            else:from warpSPHCore.polyfit import wp_polyfit as implementation
            options=dict(adjacency=adjacency,referenceParticles=rj,referenceVolumes=v[10] if volume else None,
                         sector=sector,numSectors=4,unitWeights=unit)
            if rhs:return implementation._computePolyRHS_stateBackend(qi,properties,d,exponents,v[11],subtract,**options)
            return implementation._computePolyMoments_stateBackend(qi,properties,d,exponents,**options)[0]
        return run
    return q,r,domain,props,graph,exponents,values,functions


def equation(values,domain,properties,graph,exponents,*,rhs=False,subtract=False,unit=False,sector=-1,volume=False,explicit=False):
    def ordinary(*v):
        if explicit:
            qx,rx,qh,rh,m,rho,field,vol=v[0],v[0],v[1],v[1],v[2],v[3],v[4],None
            lo,hi=domain.min.detach(),domain.max.detach()
        else:qx,rx,qh,rh,m,lo,hi,_,_,rho,vol,field=v
        dx=qx[:,None]-rx[None];length=hi-lo
        dx=torch.where(domain.periodic,(dx+length/2).remainder(length)-length/2,dx)
        selected=torch.zeros(dx.shape[:2],device=dx.device,dtype=torch.bool)
        for i in range(len(qx)):
            ids=graph.paddedJ[int(graph.edgeOffsets[i]):int(graph.edgeOffsets[i])+int(graph.numNeighbors[i])].long()
            selected[i,ids]=True
        # Excluded self pairs must not enter the independent dense oracle's
        # norm Hessian, whose unseeded 0/0 would contaminate a smooth CSR map.
        dx=torch.where(selected[...,None],dx,torch.full_like(dx,.31))
        distance=dx.norm(dim=-1);xi=-dx/qh[:,None,None]
        def kernel(h):
            z=distance/h;active=(h>0)&(z<=1)
            if properties.kernel==sph.KernelFunctions.AdhesionKernel:
                active=active&(z>.5);z=torch.where(active,z,z.new_tensor(.75))
            return torch.where(active,normalization(properties.kernel,domain.dim)*shape(z,properties.kernel,domain.dim)/h**domain.dim,0.)
        hq,hr=qh[:,None],rh[None];mode=properties.supportMode.value
        w=(kernel(hq)+kernel(hr))/2 if mode in (14,15) else kernel(hq if mode==11 else hr if mode==12 else (hq+hr)/2 if mode==13 else torch.maximum(hq,hr))
        w=(distance<qh[:,None]).to(w.dtype) if unit else w*(vol[None] if volume else (m/rho)[None])
        powers=torch.stack([xi.pow(e).prod(-1) for e in exponents],-1)
        entries=torch.tril_indices(len(exponents),len(exponents),device=exponents.device).T
        outputs=[]
        for i in range(len(qx)):
            ids=graph.paddedJ[int(graph.edgeOffsets[i]):int(graph.edgeOffsets[i])+int(graph.numNeighbors[i])].long()
            if sector>=0:
                if domain.dim==1:active=dx[i,ids,0]>0 if sector==0 else dx[i,ids,0]<0
                else:
                    angle=torch.atan2(-dx[i,ids,1],-dx[i,ids,0]).remainder(2*torch.pi)
                    active=(angle>=sector*torch.pi/2)&(angle<(sector+1)*torch.pi/2)
                active=active|(ids==i);weight=w[i,ids]*active
            else:weight=w[i,ids]
            if rhs:
                f=field[ids]-(field[i] if subtract else 0.)
                outputs.append(torch.einsum('j,ja,jc->ac',weight,powers[i,ids],f).flatten())
            else:outputs.append((weight[:,None]*powers[i,ids][:,entries[:,0]]*powers[i,ids][:,entries[:,1]]).sum(0))
        return torch.stack(outputs)
    if not explicit:return ordinary(*values)
    return torch.autograd.functional.jvp(ordinary,values[:5],values[5:],create_graph=True)[1]
