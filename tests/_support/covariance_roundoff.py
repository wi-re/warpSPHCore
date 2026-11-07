"""Independent expanded-path bounds for singular covariance boundary fixtures.

Every fixture has equal supports and one edge per row. Evaluate nonnegative
versions of the three outer products and correction polynomial, then apply
analytic magnitude envelopes for the declared kernel Jacobians. This bounds
roundoff before cancellation; it does not change smooth tolerances or NaN/Inf
matching. A separate Decimal oracle checks the cancellation cases.
"""
import torch
import warpSPHCore as sph
from .kernels import shape
from .operators import normalization


def boundary_roundoff_budgets(values,kernel,support,seed,corrected=False):
    dtype=values[0].dtype;eps=1e-7 if dtype==torch.float32 else 1e-15
    v=[x.detach().to(device='cpu',dtype=torch.float64) for x in values]
    cot=seed.detach().to(device='cpu',dtype=torch.float64).abs();n,D=v[0].shape
    x=(v[0]-v[1][:n]).abs();dt=(v[20]-v[21][:n]).abs();r=x.norm(dim=-1);h=v[2]
    if support.value==12:dh=v[23][:n].abs()
    elif support.value==13:dh=(v[22].abs()+v[23][:n].abs())/2
    elif support.value in (14,15):dh=v[22].abs()+v[23][:n].abs()
    else:dh=v[22].abs()
    active=(r>0)&(r<=h)
    if kernel==sph.KernelFunctions.AdhesionKernel:active=active&(r/h>.5)
    safe=torch.where(active,r/h,r.new_tensor(.75)).requires_grad_()
    with torch.enable_grad():
        f=shape(safe,kernel,D);f1=torch.autograd.grad(f.sum(),safe,create_graph=True)[0]
        f2=torch.autograd.grad(f1.sum(),safe,create_graph=True)[0]
        f3=torch.autograd.grad(f2.sum(),safe)[0]
    f,f1,f2,f3=[torch.where(active,z.detach().abs(),0.) for z in (f,f1,f2,f3)]
    # Literal gradient-slope compensation can restore terms lost by 1-q.
    tiny=(1-(r/h).to(dtype)==1).to(torch.float64)
    if kernel.value in (21,30,31,32):f1=f1+20*(r/h)*tiny
    C=abs(float(torch.tensor(normalization(kernel,D),dtype=dtype)))
    R=torch.where(active,r,1.);den=R.clamp_min(eps);dr=x/den[:,None]
    S=R**2+eps**2*h**2;T=R**3+eps**3*h**3;U=R+eps**2*h
    identity=torch.eye(D)[None];xx=x[:,:,None]*x[:,None,:]
    k1=C*f1/h**(D+1);k2=C*f2/h**(D+2);k3=C*f3/h**(D+3)
    H=xx/S[:,None,None]*k2[:,None,None]+(xx/T[:,None,None]+identity/U[:,None,None])*k1[:,None,None]
    G=x/den[:,None]*k1[:,None]
    W=C*f/h**D;wh=C/h**(D+1)*(D*f+(R/h)*f1)
    gh=x/den[:,None]*(C/h**(D+2)*((R/h)*f2+(D+1)*f1))[:,None]
    dW=(G*dt).sum(-1)+wh*dh;dG=torch.einsum('nce,ne->nc',H,dt)+gh*dh[:,None]
    # Positive correction equation: its Jacobian is the expanded path sum.
    def leaf(z):return z.detach().clone().requires_grad_()
    X,DX,w,dw,g,dg=[leaf(z) for z in (x,dt,W,dW,G,dG)]
    slots=(11,12,15,16,18,28,29,32,33,34)
    coefficients=[leaf(v[k].abs()) for k in slots]
    A,B,ga,gb,L,dA,dB,dga,dgb,dL=coefficients
    with torch.enable_grad():
        if corrected:
            poly=1+(B*X).sum(-1);dp=(dB*X).sum(-1)+(B*DX).sum(-1)
            cg=A[:,None]*w[:,None]*B+(A*poly)[:,None]*g+(poly*w)[:,None]*ga+(A*w)[:,None]*torch.einsum('nce,ne->nc',gb,X)
            dcg=(dA*w+A*dw)[:,None]*B+(A*w)[:,None]*dB+(dA*poly+A*dp)[:,None]*g+(A*poly)[:,None]*dg+(dp*w+poly*dw)[:,None]*ga+(poly*w)[:,None]*dga+(dA*w+A*dw)[:,None]*torch.einsum('nce,ne->nc',gb,X)+(A*w)[:,None]*(torch.einsum('nce,ne->nc',dgb,X)+torch.einsum('nce,ne->nc',gb,DX))
            cg,dcg=torch.einsum('nce,ne->nc',L,cg),torch.einsum('nce,ne->nc',dL,cg)+torch.einsum('nce,ne->nc',L,dcg)
            V=v[13][:n].abs();dV=v[30][:n].abs()
        else:
            cg,dcg=g,dg;rho=v[9][:n].abs();V=v[4][:n].abs()/rho
            dV=v[24][:n].abs()/rho+v[4][:n].abs()*v[26][:n].abs()/rho**2
        expression=dV[:,None,None]*X[:,:,None]*cg[:,None,:]+V[:,None,None]*DX[:,:,None]*cg[:,None,:]+V[:,None,None]*X[:,:,None]*dcg[:,None,:]
        leaves=[X,DX,w,dw,g,dg,*coefficients]
        bars=torch.autograd.grad(expression,leaves,cot,allow_unused=True)
    bx,bdt,bw,bdw,bg,bdg=[torch.zeros_like(z) if b is None else b for z,b in zip(leaves[:6],bars[:6])]
    # Absolute derivatives of the regularized Hessian and support gradient.
    dH=[];dgh=[]
    sup=C/h**(D+2)*((R/h)*f2+(D+1)*f1)
    sup_r=C/h**(D+3)*((R/h)*f3+(D+2)*f2)
    shifted=R+eps;normal_den=shifted**3+eps**3
    for a in range(D):
        delta=torch.zeros_like(x);delta[:,a]=1
        pair=delta[:,:,None]*x[:,None,:]+x[:,:,None]*delta[:,None,:]
        derivative=(pair/S[:,None,None]+xx*(2*R*dr[:,a]/S**2)[:,None,None])*k2[:,None,None]+xx/S[:,None,None]*(k3*dr[:,a])[:,None,None]
        derivative+=(pair/T[:,None,None]+xx*(3*R**2*dr[:,a]/T**2)[:,None,None]+identity*(dr[:,a]/U**2)[:,None,None])*k1[:,None,None]
        derivative+=(xx/T[:,None,None]+identity/U[:,None,None])*(k2*dr[:,a])[:,None,None]
        dH.append(derivative)
        normal=(delta*(shifted**2+eps**3)[:,None]+x*x[:,a,None])/normal_den[:,None]
        dgh.append(normal*sup[:,None]+x/den[:,None]*(sup_r*dr[:,a])[:,None])
    dH=torch.stack(dH,-1);dgh=torch.stack(dgh,-1)
    bx+=G*bw[:,None]+torch.einsum('nca,nc->na',H,bg+dt*bdw[:,None])
    bx+=dr*(C/h**(D+2)*((D+1)*f1+(R/h)*f2)*dh*bdw)[:,None]
    bx+=torch.einsum('ncea,ne,nc->na',dH,dt,bdg)+torch.einsum('nca,nc->na',dgh,bdg)*dh[:,None]
    bdt+=G*bdw[:,None]+torch.einsum('nca,nc->na',H,bdg)
    k1h=C/h**(D+2)*((D+1)*f1+(R/h)*f2);k2h=C/h**(D+3)*((D+2)*f2+(R/h)*f3)
    Hh=xx*(2*eps**2*h/S**2)[:,None,None]*k2[:,None,None]+xx/S[:,None,None]*k2h[:,None,None]+(xx*(3*eps**3*h**2/T**2)[:,None,None]+identity*(eps**2/U**2)[:,None,None])*k1[:,None,None]+(xx/T[:,None,None]+identity/U[:,None,None])*k1h[:,None,None]
    ghh=x/den[:,None]*(C/h**(D+3)*((D+2)*((R/h)*f2+(D+1)*f1)+(R/h)*((D+2)*f2+(R/h)*f3)))[:,None]
    whh=C/h**(D+2)*(D*(D+1)*f+2*(D+1)*(R/h)*f1+(R/h)**2*f2)
    bh=wh*bw+(gh*(bg+dt*bdw[:,None])).sum(-1)+whh*dh*bdw+(torch.einsum('nce,ne->nc',Hh,dt)*bdg).sum(-1)+(ghh*bdg).sum(-1)*dh
    bdh=wh*bdw+(gh*bdg).sum(-1)
    bounds={0:bx,20:bdt,2:bh,3:bh,22:bdh,23:bdh}
    for k,b in zip(slots,bars[6:]):
        if b is not None:bounds[k]=b
    for query,reference in ((0,1),(20,21),(2,3),(22,23)):
        target=torch.zeros_like(v[reference]);target[:n]=bounds[query];bounds[reference]=target
    machine=torch.finfo(dtype).eps;gamma=256*machine/(1-256*machine)
    # Matrix/scalar quotient pullbacks materialize denominator squared.
    # Its subnormal rounding is an absolute error, not gamma*relative error.
    # Include eta/2 in the standard |fl(z)-z| <= u|z| + eta/2 model.
    eta=float(torch.nextafter(torch.tensor(0.,dtype=dtype),torch.tensor(1.,dtype=dtype)))
    relative=[]
    for denominator in (S,T,U):
        squared=denominator**2;error=(machine/2)*squared+eta/2
        relative.append(torch.where(squared>error,error/(squared-error),0.))
    underflow=torch.stack(relative).amax(0)
    effective_gamma=gamma+underflow+gamma*underflow
    result={}
    for slot,bound in bounds.items():
        mask=active.reshape(n,*([1]*(bound.ndim-1)))
        if len(bound)!=n:mask=torch.cat((mask,torch.zeros((len(bound)-n,*mask.shape[1:]),dtype=torch.bool)))
        bound=torch.where(mask,bound,0.)
        factor=effective_gamma.reshape(n,*([1]*(bound.ndim-1)))
        if len(bound)!=n:factor=torch.cat((factor,torch.zeros((len(bound)-n,*factor.shape[1:]))))
        result[slot]=torch.nan_to_num(bound*2*factor,nan=0.,posinf=0.,neginf=0.).to(device=values[slot].device,dtype=dtype)
    return result


class DecimalJet:
    """Small independent forward derivative algebra used only by the oracles."""
    def __init__(self,value,derivative):self.v,self.d=value,tuple(derivative)
    def coerce(self,other):return other if isinstance(other,DecimalJet) else DecimalJet(other,[0]*len(self.d))
    def __add__(self,other):
        other=self.coerce(other);return DecimalJet(self.v+other.v,[a+b for a,b in zip(self.d,other.d)])
    __radd__=__add__
    def __neg__(self):return DecimalJet(-self.v,[-a for a in self.d])
    def __sub__(self,other):return self+-self.coerce(other)
    def __rsub__(self,other):return -self+other
    def __mul__(self,other):
        other=self.coerce(other);return DecimalJet(self.v*other.v,[a*other.v+self.v*b for a,b in zip(self.d,other.d)])
    __rmul__=__mul__
    def __truediv__(self,other):
        other=self.coerce(other);return DecimalJet(self.v/other.v,[(a*other.v-self.v*b)/other.v**2 for a,b in zip(self.d,other.d)])
    def __rtruediv__(self,other):return self.coerce(other)/self
    def __pow__(self,power):return DecimalJet(self.v**power,[power*self.v**(power-1)*a for a in self.d])


def decimal_oracle(values,seed,kind,support):
    """90-digit literal Spiky position/direction or viscosity coefficient bars.

    The Spiky oracle explicitly supplies the declared custom G Jacobian and
    normalized-vector Jacobian; it does not replace them by derivatives of
    their primal formulas. Hessian/support-gradient expressions are then
    differentiated in the independent Decimal algebra.
    """
    from decimal import Decimal,localcontext
    with localcontext() as context:
        context.prec=90
        number=lambda value:Decimal.from_float(float(value.detach().item() if isinstance(value,torch.Tensor) else value))
        D=values[0].shape[1];size=2*D if kind=='spiky' else 10
        result={k:[] for k in ((0,1,20,21) if kind=='spiky' else (11,12,15,16,18,28,29,32,33,34))}
        eps=number(torch.tensor(1e-7 if values[0].dtype==torch.float32 else 1e-15,dtype=values[0].dtype))
        constant=lambda v:DecimalJet(v,[0]*size)
        def variable(value,index):
            derivative=[0]*size;derivative[index]=1;return DecimalJet(value,derivative)
        for row in range(len(values[0])):
            xx=[number(values[0][row,c])-number(values[1][row,c]) for c in range(D)]
            tt=[number(values[20][row,c])-number(values[21][row,c]) for c in range(D)]
            h=number(values[2][row]);dh=number(values[22][row])
            if support.value==12:dh=number(values[23][row])
            elif support.value in (13,14,15):dh=(dh+number(values[23][row]))/2
            r=sum(z*z for z in xx).sqrt();den=max(r,eps)
            x=[variable(z,c) if kind=='spiky' else constant(z) for c,z in enumerate(xx)]
            dt=[variable(z,D+c) if kind=='spiky' else constant(z) for c,z in enumerate(tt)]
            rr=DecimalJet(r,[*(z/den for z in xx),*([0]*D)]) if kind=='spiky' else constant(r)
            q=rr/h;kernel=sph.KernelFunctions.Spiky if kind=='spiky' else sph.KernelFunctions.ViscosityKernel
            C=number(torch.tensor(normalization(kernel,D),dtype=values[0].dtype))
            if kind=='spiky':f=(1-q)**3;first=-3*(1-q)**2;second=6*(1-q)
            else:f=-q**3/2+q**2+1/(2*q)-1;first=-3*q**2/2-1/(2*q**2)+2*q;second=1/q**3-3*q+2
            k1=C*first/h**(D+1);k2=C*second/h**(D+2)
            S=rr**2+eps**2*h**2;T=rr**3+eps**3*h**3;U=rr+eps**2*h
            H=[[x[a]*x[b]/S*k2+(-x[a]*x[b]/T+(1/U if a==b else 0))*k1 for b in range(D)] for a in range(D)]
            G=[DecimalJet(xx[a]/den*k1.v,[*(H[a][c].v for c in range(D)),*([0]*D)]) if kind=='spiky' else xx[a]/den*k1 for a in range(D)]
            sup=-C/h**(D+2)*(q*second+(D+1)*first);gh=[]
            for a in range(D):
                direction=xx[a]/den
                derivative=[((((r+eps)**2+eps**3 if a==c else 0)-xx[a]*xx[c])/((r+eps)**3+eps**3)) for c in range(D)]
                normal=DecimalJet(direction,[*derivative,*([0]*D)]) if kind=='spiky' else constant(direction)
                gh.append(normal*sup)
            dG=[sum(H[a][b]*dt[b] for b in range(D))+gh[a]*dh for a in range(D)]
            if kind=='spiky':
                V=number(values[4][row])/number(values[9][row]);dV=number(values[24][row])/number(values[9][row])-number(values[4][row])*number(values[26][row])/number(values[9][row])**2
                output=constant(Decimal(0))
                for a in range(D):
                    for b in range(D):output+=(dV*(-x[a])*G[b]+V*(-dt[a])*G[b]+V*(-x[a])*dG[b])*number(seed[row,a,b])
                result[0].append(list(output.d[:D]));result[20].append(list(output.d[D:]));result[1].append([-z for z in output.d[:D]]);result[21].append([-z for z in output.d[D:]])
            else:
                slots=(11,12,15,16,18,28,29,32,33,34)
                A,B,ga,gb,L,dA,dB,dga,dgb,dL=[variable(number(values[k][row].reshape(-1)[0]),i) for i,k in enumerate(slots)]
                W=C*f/h**D;wh=-C/h**(D+1)*(D*f+q*first);dW=G[0]*dt[0]+wh*dh
                poly=1+B*x[0];dp=dB*x[0]+B*dt[0]
                cg=A*W*B+A*poly*G[0]+poly*W*ga+A*W*gb*x[0]
                dcg=(dA*W+A*dW)*B+A*W*dB+(dA*poly+A*dp)*G[0]+A*poly*dG[0]+(dp*W+poly*dW)*ga+poly*W*dga+(dA*W+A*dW)*gb*x[0]+A*W*(dgb*x[0]+gb*dt[0])
                V=number(values[13][row]);dV=number(values[30][row])
                output=(dV*(-x[0])*(L*cg)+V*(-dt[0])*(L*cg)+V*(-x[0])*(dL*cg+L*dcg))*number(seed[row,0,0])
                for k,z in zip(slots,output.d):result[k].append([z])
        if kind=='spiky':
            for k in (1,21):result[k]+=[[Decimal(0)]*D for _ in range(len(values[1])-len(values[0]))]
        return {k:[z for row in rows for z in row] for k,rows in result.items()}
