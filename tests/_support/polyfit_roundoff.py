"""Positive expanded-path bounds and 90-digit values at the viscosity pole.

Only the fixed one-edge Gather fixtures use these bounds. No smooth tolerance
or NaN/Inf comparison changes. Custom gradient reverse uses the reference's
regularized Hessian, rather than differentiating its gradient value formula.
"""
from decimal import Decimal, localcontext
import math
import torch
from .operators import normalization
from .covariance_roundoff import DecimalJet


class Magnitude:
    def __init__(self,v,b=None,d=None):self.v,self.b,self.d=v,abs(v) if b is None else b,{} if d is None else d
    def coerce(self,z):return z if isinstance(z,Magnitude) else Magnitude(float(z))
    def __add__(self,z):
        z=self.coerce(z);keys=self.d.keys()|z.d.keys()
        return Magnitude(self.v+z.v,self.b+z.b,{k:self.d.get(k,0)+z.d.get(k,0) for k in keys})
    __radd__=__add__
    def __neg__(self):return Magnitude(-self.v,self.b,self.d)
    def __sub__(self,z):return self+-self.coerce(z)
    def __rsub__(self,z):return -self+z
    def __mul__(self,z):
        z=self.coerce(z);keys=self.d.keys()|z.d.keys()
        return Magnitude(self.v*z.v,self.b*z.b,{k:self.d.get(k,0)*z.b+self.b*z.d.get(k,0) for k in keys})
    __rmul__=__mul__
    def __truediv__(self,z):
        z=self.coerce(z);den=abs(z.v);keys=self.d.keys()|z.d.keys()
        return Magnitude(self.v/z.v,self.b/den,{k:self.d.get(k,0)/den+self.b*z.d.get(k,0)/den**2 for k in keys})
    def __rtruediv__(self,z):return self.coerce(z)/self
    def __pow__(self,n):
        result=Magnitude(1.)
        for _ in range(n):result=result*self
        return result


def viscosity_fixture(values,exponents,graph,kernel,*,rhs,explicit,seed=None):
    """Return independent values, value bounds and all-input reverse bounds."""
    assert graph.numNeighbors.eq(1).all()
    dtype=values[0].dtype;D=values[0].shape[1];n=len(values[0])
    data=[z.detach().double().cpu() for z in values]
    exps=exponents.cpu().tolist();entries=torch.tril_indices(len(exps),len(exps)).T.tolist()
    C=float(torch.tensor(normalization(kernel,D),dtype=dtype));eps=1e-7 if dtype==torch.float32 else 1e-15
    machine=torch.finfo(dtype).eps;gamma=256*machine/(1-256*machine)
    exact=[];bounds=[];reverse=[torch.zeros_like(z) for z in data]
    with localcontext() as ctx:
        ctx.prec=90
        number=lambda z:Decimal.from_float(float(z))
        for i in range(n):
            j=int(graph.paddedJ[int(graph.edgeOffsets[i])]);
            def leaf(k,row,c=None):
                at=(row,) if c is None else (row,c)
                return Magnitude(float(data[k][at]),d={(k,at):1.})
            if explicit:
                x=[leaf(0,i,c)-leaf(0,j,c) for c in range(D)];h=leaf(1,i)
                dt=[leaf(5,i,c)-leaf(5,j,c) for c in range(D)];dh=leaf(6,i)
                m,rho=leaf(2,j),leaf(3,j);dm,drho=leaf(7,j),leaf(8,j)
            else:
                x=[leaf(0,i,c)-leaf(1,j,c) for c in range(D)];h=leaf(2,i)
                m,rho=leaf(4,j),leaf(9,j)
            r=math.sqrt(sum(z.v*z.v for z in x));q=r/h.v;H=h.v
            f=q**3/2+q**2+1/(2*q)+1;first=3*q*q/2+1/(2*q*q)+2*q;second=1/q**3+3*q+2
            wv=C*(-q**3/2+q*q+1/(2*q)-1)/H**D
            G=[abs(z.v)/max(r,eps)*abs(C)*first/H**(D+1) for z in x]
            wh=abs(C)/H**(D+1)*(D*f+q*first)
            wb=abs(C)*f/H**D
            wd={}
            for z,g in zip(x,G):
                for key,b in z.d.items():wd[key]=wd.get(key,0)+g*b
            for key,b in h.d.items():wd[key]=wd.get(key,0)+wh*b
            W=Magnitude(wv,wb,wd)
            V=m/rho;weight=V*W
            if explicit:
                # Gradient pullback is the declared regularized Hessian.
                HH=[[(1. if q<eps and a==b else abs(x[a].v*x[b].v)/(r*r+eps*eps*H*H))*abs(C)*second/H**(D+2)
                     +(abs(x[a].v*x[b].v)/(r**3+eps**3*H**3)+(1/(r+eps*eps*H) if a==b else 0))*abs(C)*first/H**(D+1)
                     for b in range(D)] for a in range(D)]
                gh=[abs(z.v)/max(r,eps)*abs(C)/H**(D+2)*(q*second+(D+1)*first) for z in x]
                grads=[]
                for a in range(D):
                    bars={}
                    for b,z in enumerate(x):
                        for key,bound in z.d.items():bars[key]=bars.get(key,0)+HH[a][b]*bound
                    for key,bound in h.d.items():bars[key]=bars.get(key,0)+gh[a]*bound
                    grads.append(Magnitude(x[a].v/max(r,eps)*C*(-3*q*q/2-1/(2*q*q)+2*q)/H**(D+1),G[a],bars))
                # Literal support-value expression, with clamped norm adjoint.
                rr=Magnitude(r,r,{key:sum(abs(z.v)/max(r,eps)*z.d.get(key,0) for z in x) for key in set().union(*(z.d for z in x))})
                qq=rr/h
                ff=-qq**3/2+qq**2+1/(2*qq)-1
                slope=-3*qq**2/2-1/(2*qq**2)+2*qq
                WH=-C/h**(D+2)*(D*h*ff+rr*slope)
                DW=sum(g*t for g,t in zip(grads,dt))+WH*dh
                DV=dm/rho-m*drho/(rho*rho);dweight=DV*W+V*DW
            xi=[-z/h for z in x]
            dxi=[-t/h+z*(dh/(h*h)) for z,t in zip(x,dt)] if explicit else None
            def mono(e):
                a=Magnitude(1.);da=Magnitude(0.)
                for c,power in enumerate(e):
                    term=xi[c]**power
                    if explicit:da=da*term+a*(power*xi[c]**(power-1)*dxi[c] if power else 0.)
                    a=a*term
                return a,da
            monomials=[mono(e) for e in exps];outputs=[]
            if rhs:
                for a,da in monomials:
                    for c in range(2):
                        field=leaf(4 if explicit else 11,j,c)-leaf(4 if explicit else 11,i,c)
                        if explicit:
                            df=leaf(9,j,c)-leaf(9,i,c);outputs.append((dweight*a+weight*da)*field+weight*a*df)
                        else:outputs.append(weight*a*field)
            else:
                for a,b in entries:
                    pa,da=monomials[a];pb,db=monomials[b]
                    outputs.append(dweight*pa*pb+weight*(da*pb+pa*db) if explicit else weight*pa*pb)
            bounds.append([2*gamma*z.b for z in outputs])
            if seed is not None:
                for z,cot in zip(outputs,seed[i].detach().abs().double().cpu().tolist()):
                    for (k,at),bound in z.d.items():reverse[k][at]+=2*gamma*float(cot)*bound
            # Independent high-precision value equation (no native/Warp calls).
            xx=[number(z.v) for z in x];hh=number(H);rr=sum(z*z for z in xx).sqrt();qq=rr/hh
            cc=number(C);vv=number(m.v)/number(rho.v)
            ww=cc*(-qq**3/2+qq**2+1/(2*qq)-1)/hh**D
            if explicit:
                tt=[number(z.v) for z in dt];th=number(dh.v)
                slope=-3*qq**2/2-1/(2*qq**2)+2*qq
                dw=sum(z/max(rr,number(eps))*cc*slope/hh**(D+1)*t for z,t in zip(xx,tt))
                dw-=cc/hh**(D+2)*(D*hh*(-qq**3/2+qq**2+1/(2*qq)-1)+rr*slope)*th
                dv=number(dm.v)/number(rho.v)-number(m.v)*number(drho.v)/number(rho.v)**2
                weight90=DecimalJet(vv*ww,[dv*ww+vv*dw])
                xi90=[DecimalJet(-z/hh,[-t/hh+z*th/(hh*hh)]) for z,t in zip(xx,tt)]
            else:weight90=vv*ww;xi90=[-z/hh for z in xx]
            pp=[]
            for e in exps:
                val=1
                for z,power in zip(xi90,e):
                    for _ in range(power):val=val*z
                pp.append(val)
            out=[]
            if rhs:
                for val in pp:
                    for c in range(2):
                        fv=number(data[4 if explicit else 11][j,c])-number(data[4 if explicit else 11][i,c])
                        field90=DecimalJet(fv,[number(data[9][j,c])-number(data[9][i,c])]) if explicit else fv
                        out.append(weight90*val*field90)
            else:out=[weight90*pp[a]*pp[b] for a,b in entries]
            exact.append([float(z.d[0] if explicit else z) for z in out])
    return values[0].new_tensor(exact),values[0].new_tensor(bounds),tuple(z.to(v) for z,v in zip(reverse,values))
