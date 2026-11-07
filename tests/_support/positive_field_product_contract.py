"""Positive field JVP oracle using an independently filtered ordinary operator."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .laplacian_contract import case as ordinary_case,oracle as equation


def case(device,mode,dimension=2,dtype=torch.float64,corrected=False,kernel=None,separation=None,scheme=None,field_shape=None):
    q,r,d,p,g,v,states,_=ordinary_case(device,dimension,dtype,mode=mode,kernel=kernel,field_shape=field_shape or (dimension,),scheme=scheme,positive=True)
    if separation is not None:
        eps=1e-7 if dtype==torch.float32 else 1e-15
        delta={'self':0.,'ultratiny':eps*1e-9,'clamped':eps*.85,'epsilon':eps,'near':eps*10,'cutoff':.8,'outside':.81,'interior':.56}[separation]
        d.periodic.zero_();q.kinds.zero_();r.kinds.zero_()
        g.paddedJ=torch.tensor([0,1],device=device,dtype=torch.int64);g.numNeighbors=torch.ones(2,device=device,dtype=torch.int32);g.edgeOffsets=torch.arange(2,device=device,dtype=torch.int64)
        v=list(v);v[0]=torch.zeros_like(v[0]);v[0][:,0]=delta;v[1]=torch.zeros_like(v[1]);v[2]=torch.full_like(v[2],.8);v[3]=torch.full_like(v[3],.8)
        v[14]=torch.zeros_like(v[14]);v[10]=torch.zeros_like(v[10]);v[10][0,0]=1;v[10][1,0]=-1
        v=tuple(x.detach().requires_grad_() for x in v)
    values=(*v,(v[14].detach().cos()*.03).requires_grad_(),(v[10].detach().cos()*.02).requires_grad_())
    def options(x):
        result={}
        if corrected:result.update(referenceVolumes=x[13],crkState=sph.CRKState(x[11],x[12],x[15],x[16]),gradHState=sph.GradHState(x[19],x[17]),renormalizationState=sph.RenormalizationState(x[18]))
        return result
    def product(*x):
        from warpSPHCore.native.api import warpOperationJVP
        qi,rj,domain=states(x[:10])
        return warpOperationJVP(qi,p,domain,referenceParticles=rj,adjacency=g,queryValues=x[14],referenceValues=x[10],tangentQueryValues=x[20],tangentReferenceValues=x[21],**options(x))
    def filtered(x):
        qi,rj,domain=states(x[:10]);offsets=[];counts=[];edges=[]
        with torch.no_grad():
            for i in range(len(qi.positions)):
                ids=g.paddedJ[int(g.edgeOffsets[i]):int(g.edgeOffsets[i])+int(g.numNeighbors[i])].long()
                dx=qi.positions[i]-rj.positions[ids];length=domain.max-domain.min
                dx=torch.where(domain.periodic,(dx+length/2).remainder(length)-length/2,dx)
                fi=x[14][i].reshape(-1)[:dimension]/(x[19][i] if corrected else 1)
                fj=x[10][ids].reshape(len(ids),-1)[:,:dimension]/(x[17][ids,None] if corrected else 1)
                weight=x[4][ids]*x[8][i]/x[9][ids]**2 if p.gradientMode==sph.GradientScheme.Symmetric else x[13][ids] if corrected else x[4][ids]/x[9][ids]
                keep=(dx*((fj-fi)*weight[:,None])).sum(-1)>=0
                selected=ids[keep];offsets.append(sum(counts));counts.append(len(selected));edges.append(selected)
        return sph.AdjacencyList(None,torch.cat(edges).to(sph.get_torch_index_dtype()),torch.tensor(counts,device=device,dtype=torch.int32),torch.tensor(offsets,device=device,dtype=sph.get_torch_offset_dtype()),len(qi.positions),len(rj.positions))
    def reference(backend,*x):
        from warpSPHCore.coreOperations.native.wrapper import sphOperation
        qi,rj,domain=states(x[:10])
        kwargs=options(x)
        if mode==sph.LaplacianScheme.Naive:
            # Native intentionally does not evaluate unused CRK/R corrections.
            # Warp eagerly evaluates them before selecting Naive; at singular
            # pairs its unseeded reverse can yield NaN in those unused roles.
            # Compare the consumed equation here; native ignored-role zeros
            # and their smooth derivatives are checked independently.
            kwargs.pop('crkState',None);kwargs.pop('renormalizationState',None)
        result=sphOperation(qi,replace(p,positiveDivergence=False),domain,queryValues=x[20],referenceValues=x[21],referenceParticles=rj,adjacency=filtered(x),backend=backend,**kwargs)
        packed=result.reshape(len(qi.positions),-1)
        return torch.cat((packed[:,:dimension],torch.zeros_like(packed[:,dimension:])),1).reshape_as(result)
    def smooth_equation(*x):
        data=list(x[:20]);data[14]=x[20];data[10]=x[21]
        return equation(data,d,replace(p,positiveDivergence=False),filtered(x),corrected)
    return values,product,reference,smooth_equation


def boundary_roundoff_budgets(values,kernel,seed,mode):
    """Reuse the expanded coefficient-path bound for the linear field map.

    K(theta) is the full product at zero primal fields, zero geometry/coefficient
    directions and the supplied field directions. The bound uses positive path
    sums, independently of either implementation's residual. NaN/Inf categories
    and smooth gradchecks are never relaxed.
    """
    from .laplacian_explicit_contract import boundary_roundoff_budgets as full_bounds
    from .gradient_explicit_contract import DIRECTION_SLOTS
    v=list(values[:20]);directions={k:torch.zeros_like(v[k]) for k in DIRECTION_SLOTS}
    directions[14]=values[20]/v[19][:,None]
    directions[10]=values[21]/v[17][:,None]
    v[14]=torch.zeros_like(v[14]);v[10]=torch.zeros_like(v[10])
    # This boundary fixture has one edge (i,i) per query and no periodic axes.
    x=values[0]-values[1][:len(values[0])]
    gate=(x*((values[10][:len(x)]/values[17][:len(x),None]-values[14]/values[19][:,None])*values[13][:len(x),None])).sum(-1)>=0
    seed=torch.where(gate[:,None],seed,torch.zeros_like(seed))
    if mode==sph.LaplacianScheme.Dot:
        return full_bounds((*v,*(directions[k] for k in DIRECTION_SLOTS)),kernel,sph.SupportScheme.Gather,seed,mode)
    result={}
    for component in range(seed.shape[1]):
        scalar=list(v);scalar[10]=v[10][:,component];scalar[14]=v[14][:,component]
        t=dict(directions);t[10]=directions[10][:,component];t[14]=directions[14][:,component]
        bound=full_bounds((*scalar,*(t[k] for k in DIRECTION_SLOTS)),kernel,sph.SupportScheme.Gather,seed[:,component],mode)
        for slot,value in bound.items():result[slot]=result.get(slot,torch.zeros_like(value))+value
    return result
