"""Expanded ordinary-moment bounds at the viscosity pole.

At epsilon-scale separation, x*G + W subtracts terms of order 1/r to
produce a finite result. A relative tolerance on that result cannot describe
floating-point error. Bound the positive, expanded evaluation paths instead;
keep nonfinite-category comparisons strict. The Decimal oracle
independently checks the six primal tensors at 90-digit precision.
"""
import torch
from .operators import normalization


def viscosity_path_bounds(values, kernel, seed=None, explicit=False):
    x = (values[0].detach() - values[1][:len(values[0])].detach()).abs().double().cpu()
    h = values[3][:len(x)].detach().double().cpu()
    volume = values[10][:len(x)].detach().abs().double().cpu()
    radius = x.norm(dim=-1); q = radius/h; dim = x.shape[1]
    eps = 1e-7 if values[0].dtype == torch.float32 else 1e-15
    constant = abs(float(torch.tensor(normalization(kernel, dim), dtype=values[0].dtype)))
    w = constant*(q**3/2 + q**2 + 1/(2*q) + 1)/h**dim
    g = x/radius.clamp_min(eps)[:, None]*(constant*(3*q**2/2 + 1/(2*q**2) + 2*q)/h**(dim+1))[:, None]
    identity = torch.eye(dim)[None]
    if explicit:
        assert seed is None, 'The explicit branch bounds values; reverse retains strict tolerances'
        first = 3*q**2/2 + 1/(2*q**2) + 2*q; second = 1/q**3 + 3*q + 2
        xx = x[:,:,None]*x[:,None,:]
        hessian = xx/(radius**2 + eps**2*h**2)[:,None,None]*(constant*second/h**(dim+2))[:,None,None]
        hessian += (xx/(radius**3 + eps**3*h**3)[:,None,None] + identity/(radius + eps**2*h)[:,None,None])*(constant*first/h**(dim+1))[:,None,None]
        wh = constant/h**(dim+1)*(dim*(q**3/2 + q**2 + 1/(2*q)+1) + q*first)
        gh = x/radius.clamp_min(eps)[:,None]*(constant/h**(dim+2)*(q*second + (dim+1)*first))[:,None]
        dt = (values[12].detach()-values[13][:len(x)].detach()).abs().double().cpu()
        dh = values[15][:len(x)].detach().abs().double().cpu()
        dv = values[16][:len(x)].detach().abs().double().cpu()
        dw = (g*dt).sum(-1) + wh*dh
        dg = torch.einsum('nce,ne->nc',hessian,dt) + gh*dh[:,None]
        vw, dvw = volume*w, dv*w+volume*dw
        terms = [dvw, dt*vw[:,None]+x*dvw[:,None],
            (dt[:,:,None]*x[:,None,:]+x[:,:,None]*dt[:,None,:])*vw[:,None,None] + xx*dvw[:,None,None],
            dv[:,None]*g + volume[:,None]*dg,
            dv[:,None,None]*(x[:,:,None]*g[:,None,:]+w[:,None,None]*identity)
                + volume[:,None,None]*(dt[:,:,None]*g[:,None,:]+x[:,:,None]*dg[:,None,:]+dw[:,None,None]*identity)]
        last = []
        for c in range(dim):
            for a in range(dim):
                for b in range(dim):
                    last.append(dv*(x[:,a]*x[:,b]*g[:,c]+w*(x[:,a]*(b==c)+(a==c)*x[:,b]))
                        + volume*(dt[:,a]*x[:,b]*g[:,c]+x[:,a]*dt[:,b]*g[:,c]+x[:,a]*x[:,b]*dg[:,c]
                            + dw*(x[:,a]*(b==c)+(a==c)*x[:,b])+w*(dt[:,a]*(b==c)+(a==c)*dt[:,b])))
        terms.append(torch.stack(last,-1).reshape(len(x),dim,dim,dim))
        machine = torch.finfo(values[0].dtype).eps; gamma = 256*machine/(1-256*machine)
        return tuple((2*gamma*z).to(values[0]) for z in terms)
    if seed is not None:
        x, w, g, volume = [z.detach().requires_grad_() for z in (x,w,g,volume)]
    tensors = (volume*w, x*(volume*w)[:, None],
        x[:, :, None]*x[:, None, :]*(volume*w)[:, None, None],
        volume[:, None]*g,
        volume[:, None, None]*(x[:, :, None]*g[:, None, :] + w[:, None, None]*identity),
        volume[:, None, None, None]*(g[:, :, None, None]*x[:, None, :, None]*x[:, None, None, :]
            + w[:, None, None, None]*(x[:, None, :, None]*identity[:, :, None, :]
                                      + identity[:, :, :, None]*x[:, None, None, :])))
    machine = torch.finfo(values[0].dtype).eps
    gamma = 256*machine/(1-256*machine)
    # Each implementation can round within gamma times the positive path sum.
    if seed is None: return tuple((2*gamma*z).to(values[0]) for z in tensors)
    cot = tuple(z.detach().abs().double().cpu() for z in seed)
    bx, bw, bg, bv = torch.autograd.grad(tensors, (x,w,g,volume), cot)
    first = 3*q**2/2 + 1/(2*q**2) + 2*q
    second = 1/q**3 + 3*q + 2
    k1, k2 = constant*first/h**(dim+1), constant*second/h**(dim+2)
    xx = x.detach()[:,:,None]*x.detach()[:,None,:]
    hessian = xx/(radius**2 + eps**2*h**2)[:,None,None]*k2[:,None,None]
    hessian += (xx/(radius**3 + eps**3*h**3)[:,None,None] + identity/(radius + eps**2*h)[:,None,None])*k1[:,None,None]
    wx = g.detach()
    wh = constant/h**(dim+1)*(dim*(q**3/2 + q**2 + 1/(2*q)+1) + q*first)
    gh = x.detach()/radius.clamp_min(eps)[:,None]*(constant/h**(dim+2)*(q*second + (dim+1)*first))[:,None]
    bx = bx + bw[:,None]*wx + torch.einsum('nca,nc->na',hessian,bg)
    bh = bw*wh + (bg*gh).sum(-1)
    result = {0: bx, 1: bx, 3: bh, 10: bv}
    return {k: (2*gamma*torch.cat((z, torch.zeros_like(values[k].detach().double().cpu()[len(z):])),0)).to(values[k])
            for k,z in result.items()}


def viscosity_decimal_values(values, kernel, explicit=False):
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 90
        number = lambda z: Decimal.from_float(float(z.detach() if isinstance(z, torch.Tensor) else z))
        dim = values[0].shape[1]
        eps = number(1e-7 if values[0].dtype == torch.float32 else 1e-15)
        constant = number(torch.tensor(normalization(kernel, dim), dtype=values[0].dtype))
        result = [[] for _ in range(6)]
        for i in range(len(values[0])):
            x = [number(values[0][i,c])-number(values[1][i,c]) for c in range(dim)]
            r = sum(z*z for z in x).sqrt(); h = number(values[3][i]); q = r/h
            volume = number(values[10][i]); w = constant*(-q**3/2 + q**2 + 1/(2*q)-1)/h**dim
            slope = -3*q**2/2 - 1/(2*q**2) + 2*q
            g = [z/max(r, eps)*slope*constant/h**(dim+1) for z in x]
            if explicit:
                from .covariance_roundoff import DecimalJet
                direction = [number(values[12][i,c])-number(values[13][i,c]) for c in range(dim)]
                dh = number(values[15][i]); dv = number(values[16][i])
                f = -q**3/2 + q**2 + 1/(2*q)-1
                second = 1/q**3-3*q+2
                k1, k2 = constant*slope/h**(dim+1), constant*second/h**(dim+2)
                s, t, u = r*r+eps*eps*h*h, r**3+eps**3*h**3, r+eps*eps*h
                hessian = [[x[a]*x[b]/s*k2 + (-x[a]*x[b]/t + (1/u if a==b else 0))*k1
                            for b in range(dim)] for a in range(dim)]
                gh = [-z/max(r,eps)*constant/h**(dim+2)*(q*second+(dim+1)*slope) for z in x]
                dw = sum(a*b for a,b in zip(g,direction))-constant/h**(dim+1)*(dim*f+q*slope)*dh
                dg = [sum(a*b for a,b in zip(row,direction))+z*dh for row,z in zip(hessian,gh)]
                x = [DecimalJet(z,[dz]) for z,dz in zip(x,direction)]
                w = DecimalJet(w,[dw]); g = [DecimalJet(z,[dz]) for z,dz in zip(g,dg)]
                volume = DecimalJet(volume,[dv])
            row = ([volume*w], [z*volume*w for z in x],
                [a*b*volume*w for a in x for b in x], [volume*z for z in g],
                [volume*(x[a]*g[b] + w*(a == b)) for a in range(dim) for b in range(dim)],
                [volume*(x[a]*x[b]*g[c] + w*(x[a]*(b == c) + (a == c)*x[b]))
                 for c in range(dim) for a in range(dim) for b in range(dim)])
            for target, source in zip(result, row): target.extend(float(z.d[0] if explicit else z) for z in source)
        n = len(values[0]); shapes = ((n,), (n,dim), (n,dim,dim), (n,dim), (n,dim,dim), (n,dim,dim,dim))
        return tuple(values[0].new_tensor(z).reshape(shape) for z, shape in zip(result, shapes))
