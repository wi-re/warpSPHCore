"""Full curl-product fixtures and independent smooth equations."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .curl_contract import case as curl_case,oracle as curl_equation

DIRECTION_SLOTS=(0,1,2,3,4,8,9,10,11,12,13,14,15,16,18)


def case(*args,volume=False,crk=False,renorm=False,gradh=False,geometry=True,**kwargs):
    q,r,d,p,g,v,states,_=curl_case(*args,**kwargs)
    v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
    values=(*v,*(x.detach().cos().mul(.03).requires_grad_() for x in (v[k] for k in DIRECTION_SLOTS)))
    def functions(backend,properties=p,adjacency=g):
        if backend=='native':from warpSPHCore.native.api import warpOperationJVP as compute
        else:from warpSPHCore.coreOperations.wp_curlJVP import computeSPHCurlGeometryJVP as compute
        def run(*x):
            qi,rj,domain=states(x[:10]);directions=dict(zip(DIRECTION_SLOTS,x[20:]));qt=sph.ParticleTangentState(directions[0] if geometry else None,directions[2] if geometry else None,None,directions[8]);rt=sph.ParticleTangentState(directions[1] if geometry else None,directions[3] if geometry else None,directions[4],directions[9])
            options=dict(referenceParticles=rj,referenceTangentState=rt,queryValues=x[14],referenceValues=x[10],tangentQueryValues=directions[14],tangentReferenceValues=directions[10])
            if volume:options.update(referenceVolumes=x[13],tangentReferenceVolumes=directions[13])
            if crk:options.update(crkState=sph.CRKState(x[11],x[12],x[15],x[16]),crkTangentState=sph.CRKTangentState(directions[11],directions[12],directions[15],directions[16]))
            if renorm:options.update(renormalizationState=sph.RenormalizationState(x[18]),renormalizationTangentState=sph.RenormalizationTangentState(directions[18]))
            if backend=='native':
                if gradh:options['gradHState']=sph.GradHState(x[19],x[17])
                return compute(qi,properties,domain,queryTangentState=qt,adjacency=adjacency,**options)
            if not geometry:qt=replace(qt,positions=torch.zeros_like(qi.positions))
            return compute(qi,domain,properties.kernel,properties.supportMode,adjacency,qt,gradientMode=properties.gradientMode,operationMode=properties.operationMode,**options)
        return run
    return q,r,d,p,g,values,states,functions


def oracle(values,domain,properties,graph,volume=False,crk=False,renorm=False,gradh=False,geometry=True):
    v=values[:20];directions=dict(zip(DIRECTION_SLOTS,values[20:]));directions={k:x for k,x in directions.items() if geometry or k not in (0,1,2,3)};t=tuple(directions.get(k,torch.zeros_like(x)) for k,x in enumerate(v))
    def primal(*x):
        x=list(x)
        if not volume:x[13]=x[4]/x[9]
        if not crk:
            x[11]=torch.ones_like(x[11]);x[12]=torch.zeros_like(x[12]);x[15]=torch.zeros_like(x[15]);x[16]=torch.zeros_like(x[16])
        if not renorm:x[18]=torch.eye(domain.dim,device=x[0].device,dtype=x[0].dtype).expand(len(x[0]),-1,-1)
        if not gradh:
            x[17]=torch.ones_like(x[17]);x[19]=torch.ones_like(x[19])
        return curl_equation(x,domain,properties,graph,True)
    return torch.autograd.functional.jvp(primal,v,t,create_graph=True)[1]


def boundary_case(device,dimension,dtype,kernel,support,delta,corrected=False,role='joint'):
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,support=support,volume=corrected,crk=corrected,renorm=corrected)
    d.periodic.zero_();q.kinds.zero_();r.kinds.zero_()
    g.paddedJ=torch.tensor([0,1],device=device,dtype=sph.get_torch_index_dtype());g.numNeighbors=torch.ones(2,device=device,dtype=torch.int32);g.edgeOffsets=torch.arange(2,device=device,dtype=sph.get_torch_offset_dtype())
    v=list(v);v[0]=torch.zeros_like(v[0]);v[0][:,0]=delta;v[1]=torch.zeros_like(v[1]);v[2]=torch.full_like(v[2],.8);v[3]=torch.full_like(v[3],.8)
    v[20]=torch.full_like(v[20],.03);v[21]=torch.full_like(v[21],-.02);v[22]=v[2]*.03;v[23]=v[3]*-.02
    if role!='joint':
        active=(22,23) if role=='support' else (24,25,26) if role=='mass_density' else (27,31) if role=='field' else (28,29,30,32,33,34)
        for k in range(20,35):
            if k not in active:v[k]=torch.zeros_like(v[k])
    v=tuple(x.detach().clone().requires_grad_(k not in (5,6)) for k,x in enumerate(v))
    return v,functions



def coefficient_roundoff_budgets(v,kernel,support,seed):
    from .gradient_explicit_contract import coefficient_roundoff_budgets as bound
    values=list(v)
    for k in (10,14,27,31):values[k]=values[k][...,0]
    scalar_seed=seed.reshape(len(seed))
    return bound(values,kernel,support,torch.stack((-scalar_seed,scalar_seed),1))


def viscosity_coefficient_decimal_oracle(v,seed):
    """90-digit coefficient pullbacks for the corrected 2D curl boundary."""
    from decimal import Decimal,localcontext
    from .operators import normalization
    with localcontext() as context:
        context.prec=90
        number=lambda x:Decimal.from_float(float(x.detach().item() if isinstance(x,torch.Tensor) else x))
        C=number(v[0].new_tensor(normalization(sph.KernelFunctions.ViscosityKernel,2)))
        eps=number(v[0].new_tensor(1e-7 if v[0].dtype==torch.float32 else 1e-15))
        result={16:[],29:[]}
        for i in range(len(v[0])):
            x=[number(v[0][i,c])-number(v[1][i,c]) for c in range(2)];r=abs(x[0]);h=number(v[2][i]);q=r/h
            dt=[number(v[20][i,c])-number(v[21][i,c]) for c in range(2)];dh=number(v[22][i])
            f=-q**3/2+q*q+Decimal('.5')/q-1;first=-Decimal('1.5')*q*q-Decimal('.5')/(q*q)+2*q
            W=C*f/h**2;G=[xx/max(r,eps)*(C/h**3)*first for xx in x]
            wh=-C/h**4*(2*h*f+r*first);dw=sum(a*b for a,b in zip(G,dt))+wh*dh
            V=number(v[13][i]);dV=number(v[30][i]);cot=number(seed.reshape(-1)[i])
            fi=[number(z) for z in v[14][i]];fj=[number(z) for z in v[10][i]]
            tfi=[number(z) for z in v[31][i]];tfj=[number(z) for z in v[27][i]]
            coeff=[(b-a)*V for a,b in zip(fi,fj)]
            dc=[(b-a)*dV+(tb-ta)*V for a,b,ta,tb in zip(fi,fj,tfi,tfj)]
            bg0=[dc[1]*cot,-dc[0]*cot];bdg0=[coeff[1]*cot,-coeff[0]*cot]
            bg=[sum(number(v[18][i,c,e])*bg0[c]+number(v[34][i,c,e])*bdg0[c] for c in range(2)) for e in range(2)]
            bdg=[sum(number(v[18][i,c,e])*bdg0[c] for c in range(2)) for e in range(2)]
            A=number(v[11][i]);dA=number(v[28][i]);ga=[number(z) for z in v[15][i]];F=A*W;dF=dA*W+A*dw
            result[16].extend(x[e]*(dF*bdg[c]+F*bg[c])+dt[e]*F*bdg[c] for c in range(2) for e in range(2))
            dot=A*sum(a*b for a,b in zip(G,bdg))+W*sum(a*b for a,b in zip(ga,bdg))
            result[29].extend(F*bdg[e]+x[e]*dot for e in range(2))
        return result
