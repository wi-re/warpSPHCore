"""Full gradient-product fixtures and independent smooth equations."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .gradient_contract import case as gradient_case,oracle as gradient_equation

DIRECTION_SLOTS=(0,1,2,3,4,8,9,10,11,12,13,14,15,16,18)


def case(*args,volume=False,crk=False,renorm=False,gradh=False,geometry=True,**kwargs):
    q,r,d,p,g,v,states,_=gradient_case(*args,**kwargs)
    v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
    values=(*v,*(x.detach().cos().mul(.03).requires_grad_() for x in (v[k] for k in DIRECTION_SLOTS)))
    def functions(backend,properties=p,adjacency=g):
        if backend=='native':from warpSPHCore.native.api import warpOperationJVP as compute
        else:from warpSPHCore.coreOperations.wp_gradientJVP import computeSPHGradientGeometryJVP as compute
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
        if not gradh:x[17]=torch.ones_like(x[17])
        return gradient_equation(x,domain,properties,graph,True)
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
    """Expanded-path floating-error bounds for cancellation in CRK B/gradB bars.

    The boundary fixture has scalar fields, Difference weights, equal supports
    and one reference edge per query. The nonnegative path sums bound the
    coefficient pullbacks' condition, not
    their often tiny final value. gamma(256) conservatively covers kernel
    evaluation, coefficient weights and the 1D–3D contractions. Comparing two
    implementations allows twice this bound. Nonfinite categories remain exact.
    """
    from .operators import normalization
    from .kernels import shape
    n=len(v[0]);dimension=v[0].shape[1];dtype=v[0].dtype
    x=(v[0]-v[1][:n]).detach();dt=(v[20]-v[21][:n]).detach();r=x.norm(dim=-1)
    hi=v[2].detach();hj=v[3][:n].detach();ti=v[22].detach();tj=v[23][:n].detach()
    if support.value in (14,15):h=(hi+hj)/2;dh=(ti+tj)/2;dh_abs=(ti.abs()+tj.abs())/2
    elif support.value==12:h=hj;dh=tj;dh_abs=tj.abs()
    elif support.value==13:h=(hi+hj)/2;dh=(ti+tj)/2;dh_abs=(ti.abs()+tj.abs())/2
    else:h=hi;dh=ti;dh_abs=ti.abs()
    with torch.enable_grad():
        q=(r/h).detach().requires_grad_();active=q<=1
        if kernel==sph.KernelFunctions.AdhesionKernel:active=active&(q>.5);safe=torch.where(active,q,q.new_tensor(.75))
        else:safe=q
        f=torch.where(active,shape(safe,kernel,dimension),0.)
        first=torch.autograd.grad(f.sum(),q)[0]
    with torch.no_grad():
        C=normalization(kernel,dimension);w=C*f.detach()/h**dimension
        eps=1e-7 if dtype==torch.float32 else 1e-15
        g=x/r.clamp_min(eps)[:,None]*(first*C/h**(dimension+1))[:,None]
        wh=(-C/h**(dimension+2))*((dimension*h)*f.detach()+r*first)
        volume=v[13][:n].detach();dv=v[30][:n].detach();fi=v[14].detach();fj=v[10][:n].detach();tfi=v[31].detach();tfj=v[27][:n].detach()
        coeff_bound=volume.abs()*(fi.abs()+fj.abs())
        field_bound=volume.abs()*(tfi.abs()+tfj.abs());density_bound=dv.abs()*(fi.abs()+fj.abs())
        bg_final=(field_bound+density_bound)[:,None]*seed.abs();bdg_final=coeff_bound[:,None]*seed.abs()
        L=v[18].detach().abs();dL=v[34].detach().abs()
        bg=torch.einsum('nce,nc->ne',L,bg_final)+torch.einsum('nce,nc->ne',dL,bdg_final)
        bdg=torch.einsum('nce,nc->ne',L,bdg_final)
        A=v[11].detach().abs();dA=v[28].detach().abs();ga=v[15].detach().abs()
        factor=A*w.abs();dfactor=dA*w.abs()+A*((g*dt).abs().sum(-1)+wh.abs()*dh_abs)
        gb=x.abs()[:,None,:]*(dfactor[:,None,None]*bdg[:,:,None]+factor[:,None,None]*bg[:,:,None])+dt.abs()[:,None,:]*factor[:,None,None]*bdg[:,:,None]
        tb=factor[:,None]*bdg+x.abs()*(A*(g.abs()*bdg).sum(-1)+w.abs()*(ga*bdg).sum(-1))[:,None]
        machine=torch.finfo(dtype).eps;gamma=256*machine/(1-256*machine)
        return {16:torch.nan_to_num(2*gamma*gb,nan=0.,posinf=0.,neginf=0.),29:torch.nan_to_num(2*gamma*tb,nan=0.,posinf=0.,neginf=0.)}


def viscosity_coefficient_decimal_oracle(v,seed):
    """90-digit coefficient derivatives of the 1D corrected boundary fixture."""
    from decimal import Decimal,localcontext
    from .operators import normalization
    with localcontext() as context:
        context.prec=90
        number=lambda value:Decimal.from_float(float(value.detach().item() if isinstance(value,torch.Tensor) else value))
        result={16:[],29:[]}
        C=number(v[0].new_tensor(normalization(sph.KernelFunctions.ViscosityKernel,1)).item())
        eps=number(v[0].new_tensor(1e-7 if v[0].dtype==torch.float32 else 1e-15).item())
        for i in range(len(v[0])):
            x=number(v[0][i,0])-number(v[1][i,0]);r=abs(x);h=number(v[2][i]);q=r/h
            dt=number(v[20][i,0])-number(v[21][i,0]);dh=number(v[22][i])
            f=-q**3/2+q*q+Decimal('.5')/q-1;first=-Decimal('1.5')*q*q-Decimal('.5')/(q*q)+2*q
            W=C*f/h;G=x/max(r,eps)*(C/(h*h))*first
            wh=-C/(h**3)*(h*f+r*first);dw=G*dt+wh*dh
            V=number(v[13][i]);dV=number(v[30][i]);fi=number(v[14][i]);fj=number(v[10][i]);tfi=number(v[31][i]);tfj=number(v[27][i]);cot=number(seed[i,0])
            coeff=(fj-fi)*V;dc=(fj-fi)*dV;df=(tfj-tfi)*V
            L=number(v[18][i,0,0]);dL=number(v[34][i,0,0]);bdg=L*coeff*cot;bg=L*(dc+df)*cot+dL*coeff*cot
            A=number(v[11][i]);dA=number(v[28][i]);ga=number(v[15][i,0]);F=A*W;dF=dA*W+A*dw
            result[16].append(x*(dF*bdg+F*bg)+dt*F*bdg)
            result[29].append(F*bdg+x*(A*G*bdg+W*ga*bdg))
        return result
