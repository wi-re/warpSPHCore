"""Positive path bounds for a one-edge viscosity consistency quotient.

The exact ratio is mass/volume, independent of kernel, geometry and CRK
coefficients. Taped numerator/denominator paths nevertheless subtract large
geometric bars at the viscosity pole. Bound their expanded magnitudes and
independently check the exact scalar quotient and its derivatives.
"""
import torch
from .operators import normalization


def viscosity_ratio_budgets(values,kernel,seed):
    v=[x.detach().double().cpu() for x in values];n,d=v[0].shape
    x=(v[0]-v[1][:n]).abs();r=x.norm(dim=-1);h=v[3][:n];q=r/h
    eps=1e-7 if values[0].dtype==torch.float32 else 1e-15
    c=abs(float(torch.tensor(normalization(kernel,d),dtype=values[0].dtype)))
    f=q**3/2+q**2+1/(2*q)+1;first=3*q**2/2+1/(2*q**2)+2*q
    w=c*f/h**d;wx=x/r.clamp_min(eps)[:,None]*(c*first/h**(d+1))[:,None]
    wh=c/h**(d+1)*(d*f+q*first)
    A=v[12].abs();B=v[13].abs();poly=1+(B*x).sum(-1)
    # Use actual finite denominator magnitudes, not an already cancelling
    # geometric bar, to expand the two quotient paths independently.
    raw=c*(-q**3/2+q**2+1/(2*q)-1)/h**d
    signed_poly=1+((v[0]-v[1][:n])*v[13]).sum(-1)
    corrected=v[12]*signed_poly*raw;volume=v[10][:n].abs();mass=v[4][:n].abs()
    denominator=(v[10][:n]**2*corrected).abs()
    bnum=seed.detach().abs().double().cpu()/denominator
    bden=bnum*(mass/volume)
    bw=bnum*mass*volume+bden*volume**2
    bx=bw[:,None]*(A[:,None]*B*w[:,None]+(A*poly)[:,None]*wx)
    bh=bw*A*poly*wh
    result={0:bx,1:bx,3:bh,4:bnum*volume*A*poly*w,
        10:(bnum*mass+2*bden*volume)*A*poly*w,12:bw*poly*w,13:bw[:,None]*(A*w)[:,None]*x}
    u=torch.finfo(values[0].dtype).eps;gamma=256*u/(1-256*u)
    return {k:(2*gamma*torch.cat((z,torch.zeros_like(v[k][len(z):])),0)).to(values[k]) for k,z in result.items()}


def exact_one_edge_bars(values,seed):
    from decimal import Decimal,localcontext
    result=[torch.zeros_like(x) for x in values]
    with localcontext() as ctx:
        ctx.prec=90
        number=lambda z:Decimal.from_float(float(z.detach()))
        for row in range(len(values[0])):
            m,V,s=number(values[4][row]),number(values[10][row]),number(seed[row])
            result[4][row]=float(s/V);result[10][row]=float(-s*m/V**2)
    return result
