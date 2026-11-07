"""Complete covariance explicit products with an independent smooth equation."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .covariance_contract import case as covariance_case,oracle as covariance_equation
from .gradient_explicit_contract import DIRECTION_SLOTS


def case(*args,volume=False,crk=False,renorm=False,geometry=True,**kwargs):
    q,r,d,p,g,v,states,_=covariance_case(*args,**kwargs)
    v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
    values=(*v,*(x.detach().cos().mul(.03).requires_grad_() for x in (v[k] for k in DIRECTION_SLOTS)))
    def functions(backend,properties=p,adjacency=g,public=False):
        if backend=='native':from warpSPHCore.native.api import warpOperationJVP as compute
        elif backend=='warp_extension':from .covariance_extension_reference import computeCovarianceGeometryJVP as compute
        elif backend=='warp_gradient':from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP as compute
        elif public:compute=sph.warpOperationJVP
        else:from warpSPHCore.coreOperations.wp_covarianceJVP import computeCovarianceGeometryJVP as compute
        def run(*x):
            qi,rj,domain=states(x[:10]);directions=dict(zip(DIRECTION_SLOTS,x[20:]))
            qt=sph.ParticleTangentState(directions[0] if geometry else None,directions[2] if geometry else None,None,directions[8])
            rt=sph.ParticleTangentState(directions[1] if geometry else None,directions[3] if geometry else None,directions[4],directions[9])
            options=dict(referenceParticles=rj,referenceTangentState=rt)
            if volume:options.update(referenceVolumes=x[13],tangentReferenceVolumes=directions[13])
            if crk:options.update(crkState=sph.CRKState(x[11],x[12],x[15],x[16]),crkTangentState=sph.CRKTangentState(directions[11],directions[12],directions[15],directions[16]))
            if renorm:options.update(renormalizationState=sph.RenormalizationState(x[18]),renormalizationTangentState=sph.RenormalizationTangentState(directions[18]))
            if backend=='warp_gradient':
                if not geometry:qt=replace(qt,positions=torch.zeros_like(qi.positions))
                result=[]
                for coordinate in range(domain.dim):
                    fields=(qi.positions[:,coordinate],rj.positions[:,coordinate],directions[0][:,coordinate] if geometry else torch.zeros_like(qi.supports),directions[1][:,coordinate] if geometry else torch.zeros_like(rj.supports))
                    options.update(zip(('queryValues','referenceValues','tangentQueryValues','tangentReferenceValues'),fields))
                    result.append(compute(qi,domain,properties.kernel,properties.supportMode,adjacency,qt,gradientMode=sph.GradientScheme.Difference,operationMode=properties.operationMode,**options))
                return torch.stack(result,dim=1)
            if backend=='native' or public:return compute(qi,properties,domain,queryTangentState=qt,adjacency=adjacency,**options)
            if not geometry:qt=replace(qt,positions=torch.zeros_like(qi.positions))
            return compute(qi,domain,properties.kernel,properties.supportMode,adjacency,qt,operationMode=properties.operationMode,**options)
        return run
    return q,r,d,p,g,values,states,functions


def oracle(values,domain,properties,graph,volume=False,crk=False,renorm=False,geometry=True):
    v=values[:20];directions=dict(zip(DIRECTION_SLOTS,values[20:]));directions={k:x for k,x in directions.items() if (geometry or k not in (0,1,2,3)) and k not in (10,14)}
    tangent=tuple(directions.get(k,torch.zeros_like(x)) for k,x in enumerate(v))
    def primal(*x):
        x=list(x)
        if not volume:x[13]=x[4]/x[9]
        if not crk:
            x[11]=torch.ones_like(x[11]);x[12]=torch.zeros_like(x[12]);x[15]=torch.zeros_like(x[15]);x[16]=torch.zeros_like(x[16])
        if not renorm:x[18]=torch.eye(domain.dim,device=x[0].device,dtype=x[0].dtype).expand(len(x[0]),-1,-1)
        return covariance_equation(x,domain,properties,graph,True)
    return torch.autograd.functional.jvp(primal,v,tangent,create_graph=True)[1]


def boundary_case(device,dimension,dtype,kernel,support,delta,corrected=False,role='joint'):
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,support=support,volume=corrected,crk=corrected,renorm=corrected)
    d.periodic.zero_();q.kinds.zero_();r.kinds.zero_()
    g.paddedJ=torch.tensor([0,1],device=device,dtype=sph.get_torch_index_dtype());g.numNeighbors=torch.ones(2,device=device,dtype=torch.int32);g.edgeOffsets=torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype())
    v=list(v);v[0]=torch.zeros_like(v[0]);v[0][:,0]=delta;v[1]=torch.zeros_like(v[1]);v[2]=torch.full_like(v[2],.8);v[3]=torch.full_like(v[3],.8)
    v[20]=torch.full_like(v[20],.03);v[21]=torch.full_like(v[21],-.02);v[22]=v[2]*.03;v[23]=v[3]*-.02
    if role!='joint':
        active=(22,23) if role=='support' else (24,25,26) if role=='mass_density' else (28,29,30,32,33,34)
        for k in range(20,35):
            if k not in active:v[k]=torch.zeros_like(v[k])
    v=tuple(x.detach().clone().requires_grad_(k not in (5,6)) for k,x in enumerate(v))
    return v,functions
