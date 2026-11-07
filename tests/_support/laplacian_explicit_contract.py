"""Full Laplacian-product fixtures and independent smooth equations."""
from dataclasses import replace
import torch
import warpSPHCore as sph
from .laplacian_contract import case as laplacian_case,oracle as laplacian_equation

DIRECTION_SLOTS=(0,1,2,3,4,8,9,10,11,12,13,14,15,16,18)


def case(*args,volume=False,crk=False,renorm=False,gradh=False,geometry=True,**kwargs):
    q,r,d,p,g,v,states,_=laplacian_case(*args,**kwargs)
    v=list(v);v[5]=v[5].detach();v[6]=v[6].detach();v=tuple(v)
    values=(*v,*(x.detach().cos().mul(.03).requires_grad_() for x in (v[k] for k in DIRECTION_SLOTS)))
    def functions(backend,properties=p,adjacency=g):
        if backend=='native':from warpSPHCore.native.api import warpOperationJVP as compute
        else:from warpSPHCore.coreOperations.wp_laplacianJVP import computeSPHLaplacianGeometryJVP as compute
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
            return compute(qi,domain,properties.kernel,properties.supportMode,adjacency,qt,gradientMode=properties.gradientMode,operationMode=properties.operationMode,laplacianMode=properties.laplacianMode,**options)
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
        return laplacian_equation(x,domain,properties,graph,True)
    return torch.autograd.functional.jvp(primal,v,t,create_graph=True)[1]


def boundary_case(device,dimension,dtype,kernel,support,delta,corrected=False,mode=None,role='joint'):
    q,r,d,p,g,v,states,functions=case(device,dimension,dtype,kernel=kernel,support=support,mode=mode,field_shape=(dimension,) if mode==sph.LaplacianScheme.Dot else (),volume=corrected,crk=corrected,renorm=corrected)
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



def boundary_roundoff_budgets(v,kernel,support,seed,mode=sph.LaplacianScheme.Brookshaw):
    """Expanded-path bounds for correction and Dot-direction cancellation.

    The boundary fixture has scalar/vector fields, Difference weights, equal supports
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
        first=torch.autograd.grad(f.sum(),q,create_graph=True)[0]
        second=torch.autograd.grad(first.sum(),q)[0]
    with torch.no_grad():
        C=normalization(kernel,dimension);w=C*f.detach()/h**dimension
        eps=1e-7 if dtype==torch.float32 else 1e-15
        g=x/r.clamp_min(eps)[:,None]*(first*C/h**(dimension+1))[:,None]
        wh=(-C/h**(dimension+2))*((dimension*h)*f.detach()+r*first)
        volume=v[13][:n].detach();dv=v[30][:n].detach();fi=v[14].detach();fj=v[10][:n].detach();tfi=v[31].detach();tfj=v[27][:n].detach()
        if fi.ndim>1:volume=volume[:,None];dv=dv[:,None]
        coeff_bound=volume.abs()*(fi.abs()+fj.abs())
        field_bound=volume.abs()*(tfi.abs()+tfj.abs());density_bound=dv.abs()*(fi.abs()+fj.abs())
        # The geometry-chain support uses max for codes 14/15, whereas the
        # raw G/dG kernels average those supports. At the equal-support tie the
        # geometry chain takes the query direction.
        if support.value==12:chain_h=hj;chain_dh=tj;chain_abs=tj.abs()
        elif support.value==13:chain_h=(hi+hj)/2;chain_dh=(ti+tj)/2;chain_abs=(ti.abs()+tj.abs())/2
        else:chain_h=hi;chain_dh=ti;chain_abs=ti.abs()
        divisor=r+1e-8*chain_h
        normal=x/divisor[:,None]
        dr_bound=(x.abs()*dt.abs()).sum(-1)/r
        dd_bound=dr_bound+1e-8*chain_abs
        dn_bound=(dt.abs()+normal.abs()*dd_bound[:,None])/divisor[:,None]
        if mode==sph.LaplacianScheme.Dot:
            projection=(coeff_bound*normal.abs()).sum(-1)
            dprojection=((field_bound+density_bound)*normal.abs()+coeff_bound*dn_bound).sum(-1)
            left=(dimension+2)*projection[:,None]*normal.abs()
            dleft=(dimension+2)*(dprojection[:,None]*normal.abs()+projection[:,None]*dn_bound)
            bp=(seed.abs()*(dleft+field_bound+density_bound)).sum(-1)
            bdp=(seed.abs()*(left+coeff_bound)).sum(-1)
            bg_final=bp[:,None]*normal.abs()/divisor[:,None]+bdp[:,None]*(dn_bound/divisor[:,None]+normal.abs()*dd_bound[:,None]/divisor[:,None]**2)
            bdg_final=bdp[:,None]*normal.abs()/divisor[:,None]
        elif mode==sph.LaplacianScheme.Default:
            divisor2=r+1e-12*chain_h
            dd_bound2=dr_bound+1e-12*chain_abs
            normal2=normal/divisor2[:,None]
            dn_bound2=(dn_bound+normal2.abs()*dd_bound2[:,None])/divisor2[:,None]
            bg_final=2*seed.abs().reshape(n,1)*((field_bound+density_bound)[:,None]*normal2.abs()+coeff_bound[:,None]*dn_bound2)
            bdg_final=2*seed.abs().reshape(n,1)*coeff_bound[:,None]*normal2.abs()
        else:
            bg_final=2*seed.abs().reshape(n,1)*((field_bound+density_bound)[:,None]*normal.abs()/divisor[:,None]+coeff_bound[:,None]*dn_bound/divisor[:,None]+coeff_bound[:,None]*normal.abs()*dd_bound[:,None]/divisor[:,None]**2)
            bdg_final=2*seed.abs().reshape(n,1)*coeff_bound[:,None]*normal.abs()/divisor[:,None]
        L=v[18].detach().abs();dL=v[34].detach().abs()
        bg=torch.einsum('nce,nc->ne',L,bg_final)+torch.einsum('nce,nc->ne',dL,bdg_final)
        bdg=torch.einsum('nce,nc->ne',L,bdg_final)
        A=v[11].detach().abs();dA=v[28].detach().abs();ga=v[15].detach().abs()
        factor=A*w.abs();dfactor=dA*w.abs()+A*((g*dt).abs().sum(-1)+wh.abs()*dh_abs)
        gb=x.abs()[:,None,:]*(dfactor[:,None,None]*bdg[:,:,None]+factor[:,None,None]*bg[:,:,None])+dt.abs()[:,None,:]*factor[:,None,None]*bdg[:,:,None]
        tb=factor[:,None]*bdg+x.abs()*(A*(g.abs()*bdg).sum(-1)+w.abs()*(ga*bdg).sum(-1))[:,None]
        # Expand the remaining coefficient paths too: Dot can cancel the
        # leading transverse B bar near a singular kernel.
        eye=torch.eye(dimension,device=x.device,dtype=dtype)[None]
        outer=x.abs()[:,:,None]*x.abs()[:,None,:]
        hessian=outer/(r*r+eps*eps*h*h)[:,None,None]*(C*second/h**(dimension+2)).abs()[:,None,None]
        hessian+=(outer/(r**3+eps**3*h**3)[:,None,None]+eye/(r+eps*eps*h)[:,None,None])*(C*first/h**(dimension+1)).abs()[:,None,None]
        gh=x.abs()/r.clamp_min(eps)[:,None]*(abs(C)/h**(dimension+2)*(q.abs()*second.abs()+(dimension+1)*first.abs()))[:,None]
        dg=torch.einsum('nce,ne->nc',hessian,dt.abs())+gh*dh_abs[:,None]
        dw=(g*dt).abs().sum(-1)+wh.abs()*dh_abs
        B=v[12].detach().abs();dB=v[29].detach().abs();dga=v[32].detach().abs();gbp=v[16].detach().abs();dgb=v[33].detach().abs()
        poly=1+(B*x.abs()).sum(-1);dp=(dB*x.abs()+B*dt.abs()).sum(-1)
        dot=lambda u,z:(u*z).sum(-1)
        bb=factor[:,None]*bg+x.abs()*(A*dot(g.abs(),bg)+w.abs()*dot(ga,bg))[:,None]
        bb+=dfactor[:,None]*bdg+(dA[:,None]*x.abs()+A[:,None]*dt.abs())*dot(g.abs(),bdg)[:,None]
        bb+=x.abs()*(A*dot(dg,bdg)+dw*dot(ga,bdg)+w.abs()*dot(dga,bdg))[:,None]+dt.abs()*(w.abs()*dot(ga,bdg))[:,None]
        gba=(poly*w.abs())[:,None]*bg+(dp*w.abs()+poly*dw)[:,None]*bdg
        tdga=(poly*w.abs())[:,None]*bdg
        gx=torch.einsum('nce,ne->nc',gbp,x.abs());dgx=torch.einsum('nce,ne->nc',dgb,x.abs())+torch.einsum('nce,ne->nc',gbp,dt.abs())
        a_path=w.abs()[:,None]*B+poly[:,None]*g.abs()+w.abs()[:,None]*gx
        da_path=dw[:,None]*B+w.abs()[:,None]*dB+dp[:,None]*g.abs()+poly[:,None]*dg+dw[:,None]*gx+w.abs()[:,None]*dgx
        ba=dot(a_path,bg)+dot(da_path,bdg);tba=dot(a_path,bdg)
        corrected=A[:,None]*w.abs()[:,None]*B+A[:,None]*poly[:,None]*g.abs()+poly[:,None]*w.abs()[:,None]*ga+A[:,None]*w.abs()[:,None]*gx
        dcorrected=dfactor[:,None]*B+factor[:,None]*dB+(dA*poly+A*dp)[:,None]*g.abs()+(A*poly)[:,None]*dg+(dp*w.abs()+poly*dw)[:,None]*ga+(poly*w.abs())[:,None]*dga+dfactor[:,None]*gx+factor[:,None]*dgx
        br=bg_final[:,:,None]*corrected[:,None,:]+bdg_final[:,:,None]*dcorrected[:,None,:]
        tbr=bdg_final[:,:,None]*corrected[:,None,:]
        machine=torch.finfo(dtype).eps;gamma=256*machine/(1-256*machine)
        result={11:ba,12:bb,15:gba,16:gb,18:br,28:tba,29:tb,32:tdga,33:factor[:,None,None]*x.abs()[:,None,:]*bdg[:,:,None],34:tbr}
        budgets={slot:torch.where((r>0).reshape(n,*([1]*(bound.ndim-1))),torch.nan_to_num(2*gamma*bound,nan=0.,posinf=0.,neginf=0.),0.) for slot,bound in result.items()}
        if mode==sph.LaplacianScheme.Dot:
            # The product is linear in position directions. Its direction
            # Jacobian is a sum of normal/projection and corrected-gradient
            # paths that can almost cancel. Bound those paths before summing.
            unit=x.abs()/r[:,None]
            normal_jac=(eye+normal.abs()[:,:,None]*unit[:,None,:])/divisor[:,None,None]
            kc=A[:,None,None]*(B[:,:,None]*g.abs()[:,None,:]+g.abs()[:,:,None]*B[:,None,:]+poly[:,None,None]*hessian)
            kc+=w.abs()[:,None,None]*ga[:,:,None]*B[:,None,:]+poly[:,None,None]*ga[:,:,None]*g.abs()[:,None,:]
            kc+=A[:,None,None]*gx[:,:,None]*g.abs()[:,None,:]+factor[:,None,None]*gbp
            corrected_jac=torch.einsum('nce,nef->ncf',L,kc)
            transpose=lambda matrix,vector:torch.einsum('nce,nc->ne',matrix,vector)
            dot_gn=(corrected*normal.abs()).sum(-1)
            phi=dot_gn/divisor
            normal_part=(dimension+2)*phi[:,None]*((seed.abs()*normal.abs()).sum(-1)[:,None]*transpose(normal_jac,coeff_bound)+projection[:,None]*transpose(normal_jac,seed.abs()))
            gradient_part=bdp[:,None]*((transpose(corrected_jac,normal.abs())+transpose(normal_jac,corrected))/divisor[:,None]+dot_gn[:,None]*unit/divisor[:,None]**2)
            bound=torch.where((r>0)[:,None],torch.nan_to_num(2*gamma*(normal_part+gradient_part),nan=0.,posinf=0.,neginf=0.),0.)
            budgets[20]=bound;budgets[21]=torch.zeros_like(v[21]);budgets[21][:n]=bound
        return budgets



def viscosity_coefficient_decimal_oracle(v,seed,mode=sph.LaplacianScheme.Brookshaw):
    """90-digit corrected-estimator coefficient derivatives of the 1D corrected boundary fixture."""
    from decimal import Decimal,localcontext
    from .operators import normalization
    with localcontext() as context:
        context.prec=90
        number=lambda value:Decimal.from_float(float(value.detach().item() if isinstance(value,torch.Tensor) else value))
        result={k:[] for k in (11,12,15,16,18,28,29,32,33,34)}
        C=number(v[0].new_tensor(normalization(sph.KernelFunctions.ViscosityKernel,1)).item())
        eps=number(v[0].new_tensor(1e-7 if v[0].dtype==torch.float32 else 1e-15).item())
        for i in range(len(v[0])):
            x=number(v[0][i,0])-number(v[1][i,0]);r=abs(x);h=number(v[2][i]);q=r/h
            dt=number(v[20][i,0])-number(v[21][i,0]);dh=number(v[22][i])
            f=-q**3/2+q*q+Decimal('.5')/q-1;first=-Decimal('1.5')*q*q-Decimal('.5')/(q*q)+2*q
            W=C*f/h;G=x/max(r,eps)*(C/(h*h))*first
            wh=-C/(h**3)*(h*f+r*first);dw=G*dt+wh*dh
            V=number(v[13][i]);dV=number(v[30][i]);fi=number(v[14][i]);fj=number(v[10][i]);tfi=number(v[31][i]);tfj=number(v[27][i]);cot=number(seed.flatten()[i])
            coeff=(fj-fi)*V;dc=(fj-fi)*dV;df=(tfj-tfi)*V
            den=r+number(v[0].new_tensor(1e-8))*h;normal=x/den
            dr=x*dt/r;dden=dr+number(v[0].new_tensor(1e-8))*dh;dn=(dt-normal*dden)/den
            if mode==sph.LaplacianScheme.Default:
                den2=r+number(v[0].new_tensor(1e-12))*h;dden2=dr+number(v[0].new_tensor(1e-12))*dh
                normal2=normal/den2;dn2=(dn-normal2*dden2)/den2
                bdg_final=-2*coeff*normal2*cot
                bg_final=-2*cot*((dc+df)*normal2+coeff*dn2)
            elif mode==sph.LaplacianScheme.Dot:
                projection=coeff*normal;dprojection=(dc+df)*normal+coeff*dn
                left=3*projection*normal;dleft=3*(dprojection*normal+projection*dn)
                bp=cot*(-dleft+dc+df);bdp=cot*(-left+coeff)
                bdg_final=bdp*normal/den
                bg_final=bp*normal/den+bdp*(dn/den-normal*dden/(den*den))
            else:
                bdg_final=-2*coeff*normal*cot/den
                bg_final=-2*cot*((dc+df)*normal/den+coeff*dn/den-coeff*normal*dden/(den*den))
            L=number(v[18][i,0,0]);dL=number(v[34][i,0,0]);bdg=L*bdg_final;bg=L*bg_final+dL*bdg_final
            A=number(v[11][i]);dA=number(v[28][i]);ga=number(v[15][i,0]);F=A*W;dF=dA*W+A*dw
            second=1/(q**3)-3*q+2
            k1=C*first/(h*h);k2=C*second/(h**3)
            H=x*x/(r*r+eps*eps*h*h)*k2+(-x*x/(r**3+eps**3*h**3)+1/(r+eps*eps*h))*k1
            gh=x/max(r,eps)*(-C/(h**3))*(q*second+2*first)
            dg=H*dt+gh*dh;dga=number(v[32][i,0])
            B=number(v[12][i,0]);dB=number(v[29][i,0]);gb=number(v[16][i,0,0]);dgb=number(v[33][i,0,0])
            poly=1+B*x;dp=dB*x+B*dt;gx=gb*x;dgx=dgb*x+gb*dt
            a_path=W*B+poly*G+W*gx
            da_path=dw*B+W*dB+dp*G+poly*dg+dw*gx+W*dgx
            corrected=A*W*B+A*poly*G+poly*W*ga+A*W*gx
            dcorrected=dF*B+F*dB+(dA*poly+A*dp)*G+A*poly*dg+(dp*W+poly*dw)*ga+poly*W*dga+dF*gx+F*dgx
            result[11].append(a_path*bg+da_path*bdg)
            result[15].append(poly*W*bg+(dp*W+poly*dw)*bdg)
            result[18].append(bg_final*corrected+bdg_final*dcorrected)
            result[28].append(a_path*bdg)
            result[32].append(poly*W*bdg)
            result[33].append(F*x*bdg)
            result[34].append(bdg_final*corrected)
            result[12].append(F*bg+x*(A*G+W*ga)*bg+dF*bdg+(dA*x+A*dt)*G*bdg+A*x*dg*bdg+(dt*W+x*dw)*ga*bdg+x*W*dga*bdg)
            result[16].append(x*(dF*bdg+F*bg)+dt*F*bdg)
            result[29].append(F*bdg+x*(A*G*bdg+W*ga*bdg))
        return result


def dot_direction_decimal_oracle(v,seed,kernel):
    """90-digit Jacobian of the literal Dot JVP in its position directions.

    Independent normal/projection differentiation and the declared custom
    kernel Hessian give d(product)/d(dx). Correction directions do not enter
    this Jacobian because the full product is linear in all input directions.
    The fixture uses Difference weights, supplied volumes, CRK and renorm,
    Gather support, one edge per query and an unmasked nonzero radius.
    """
    from decimal import Decimal,localcontext
    from .operators import normalization
    with localcontext() as context:
        context.prec=90
        number=lambda value:Decimal.from_float(float(value.detach().item() if isinstance(value,torch.Tensor) else value))
        d=v[0].shape[1];rows=[]
        zero=Decimal(0);one=Decimal(1)
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),zero)
        mv=lambda m,x:[dot(row,x) for row in m]
        C=number(v[0].new_tensor(normalization(kernel,d)))
        eps=number(v[0].new_tensor(1e-7 if v[0].dtype==torch.float32 else 1e-15))
        distance_eps=number(v[0].new_tensor(1e-8))
        for i in range(len(v[0])):
            x=[number(v[0][i,c])-number(v[1][i,c]) for c in range(d)]
            radius=dot(x,x).sqrt();h=number(v[2][i]);ratio=radius/h
            if kernel==sph.KernelFunctions.CubicSpline:
                f=(1-ratio)**3;first=-3*(1-ratio)**2;second=6*(1-ratio)
                if ratio<Decimal('.5'):
                    f-=4*(Decimal('.5')-ratio)**3
                    first+=12*(Decimal('.5')-ratio)**2
                    second-=24*(Decimal('.5')-ratio)
            elif kernel==sph.KernelFunctions.ViscosityKernel:
                f=-ratio**3/2+ratio*ratio+Decimal('.5')/ratio-1
                first=-Decimal('1.5')*ratio*ratio-Decimal('.5')/(ratio*ratio)+2*ratio
                second=1/(ratio**3)-3*ratio+2
            else:raise AssertionError(kernel)
            W=C*f/h**d;k1=C*first/h**(d+1);k2=C*second/h**(d+2)
            g=[a/max(radius,eps)*k1 for a in x]
            H=[[(x[c]*x[e]/(radius*radius+eps*eps*h*h))*k2+(-x[c]*x[e]/(radius**3+eps**3*h**3)+(one if c==e else zero)/(radius+eps*eps*h))*k1 for e in range(d)] for c in range(d)]
            A=number(v[11][i]);B=[number(z) for z in v[12][i]];ga=[number(z) for z in v[15][i]]
            gb=[[number(z) for z in row] for row in v[16][i]];R=[[number(z) for z in row] for row in v[18][i]]
            poly=1+dot(B,x);gx=mv(gb,x)
            corrected=[A*W*B[c]+A*poly*g[c]+poly*W*ga[c]+A*W*gx[c] for c in range(d)]
            kc=[[A*B[c]*g[e]+A*B[e]*g[c]+A*poly*H[c][e]+B[e]*W*ga[c]+poly*g[e]*ga[c]+A*g[e]*gx[c]+A*W*gb[c][e] for e in range(d)] for c in range(d)]
            G=mv(R,corrected);K=[[sum((R[c][a]*kc[a][e] for a in range(d)),zero) for e in range(d)] for c in range(d)]
            den=radius+distance_eps*h;normal=[a/den for a in x];unit=[a/radius for a in x]
            N=[[( (one if c==e else zero)-normal[c]*unit[e])/den for e in range(d)] for c in range(d)]
            V=number(v[13][i]);q=[V*(number(v[10][i,c])-number(v[14][i,c])) for c in range(d)];cot=[number(z) for z in seed[i]]
            projection=dot(q,normal);gn=dot(G,normal);P=gn/den
            left=[(d+2)*projection*a for a in normal];beta=dot(cot,[q[c]-left[c] for c in range(d)])
            row=[]
            for e in range(d):
                grad_p=(sum((K[c][e]*normal[c]+N[c][e]*G[c] for c in range(d)),zero))/den-gn/(den*den)*unit[e]
                normal_path=-(d+2)*P*(dot(cot,normal)*sum((q[c]*N[c][e] for c in range(d)),zero)+projection*sum((cot[c]*N[c][e] for c in range(d)),zero))
                row.append(normal_path+beta*grad_p)
            rows.extend(row)
        reference=[-x for x in rows]+[zero]*((len(v[1])-len(v[0]))*d)
        return {20:rows,21:reference}
