#!/usr/bin/env python3
"""Frozen production PolyFit advection; correctness/conditioning, no speed gate."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--precision', choices=['float32','float64'], default='float64')
    parser.add_argument('--n', type=int, nargs='+', default=[24,48,96])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['warpSPHCore_PRECISION'] = args.precision
    import torch
    import warp as wp
    from warpSPHCore import ParticleState, DomainDescription, KernelFunctions, radiusSearchCompactHashMap
    from warpSPHCore.polyfit import PolyFit
    wp.init()
    torch.set_grad_enabled(False)
    dtype = getattr(torch,args.precision)
    rows = []
    for constant in (True,False):
        for n in args.n:
            k = torch.arange(n,dtype=dtype,device=args.device)
            x = (k+.5+.12*(7.3*k).sin())/n
            volume = ((x.roll(-1)-x)%1 + (x-x.roll(1))%1)*.5
            p = ParticleState(positions=x[:,None],supports=torch.full_like(x,5.5/n),masses=volume,
                densities=torch.ones_like(x),kinds=torch.zeros(n,dtype=torch.int32,device=args.device))
            dom = DomainDescription(x.new_zeros(1),x.new_ones(1),torch.ones(1,dtype=torch.bool,device=args.device),1)
            adj = radiusSearchCompactHashMap(p,dom)
            fit = PolyFit.build(p,dom,KernelFunctions.Wendland2,3,constant=constant,adjacency=adj)
            if bool(fit.deficient.any()):
                raise AssertionError('Manufactured periodic ladder needs full-rank fits.')
            initial = (2*math.pi*x).sin()
            derivative = fit.gradient(initial)[:,0]
            exact_derivative = 2*math.pi*(2*math.pi*x).cos()
            u = initial.clone()
            time, steps, end = 0.,0,.05
            rhs = lambda values: -fit.gradient(values)[:,0]
            while time < end-1e-14:
                dt = min(.15/n,end-time)
                k1 = rhs(u)
                k2 = rhs(u+.5*dt*k1)
                k3 = rhs(u+.5*dt*k2)
                k4 = rhs(u+dt*k3)
                u = u+dt*(k1+2*k2+2*k3+k4)/6
                time += dt
                steps += 1
            exact = (2*math.pi*(x-end)).sin()
            error = u-exact
            row = dict(n=n,constant=constant,order=3,jitter_amplitude=.12,steps=steps,time=end,
                deficient=int(fit.deficient.sum()),condition_max=float(fit.cond.max()),
                derivative_l2=float((volume*(derivative-exact_derivative).square()).sum().sqrt()),
                evolution_l2=float((volume*error.square()).sum().sqrt()),
                evolution_linf=float(error.abs().max()),weighted_mass_drift=float((volume*(u-initial)).sum()),
                exact_sample_mass_drift=float((volume*(exact-initial)).sum()))
            if not torch.isfinite(u).all() or row['evolution_linf']>.02:
                raise AssertionError(row)
            rows.append(row)
    rates = []
    for constant in (True,False):
        ladder = [r for r in rows if r['constant']==constant]
        for coarse,fine in zip(ladder,ladder[1:]):
            rate = math.log(coarse['evolution_l2']/fine['evolution_l2'])/math.log(fine['n']/coarse['n'])
            rates.append(dict(constant=constant,coarse=coarse['n'],fine=fine['n'],evolution_order=rate))
            if args.precision=='float64' and rate<2.5:
                raise AssertionError(rates[-1])
    root = Path(__file__).resolve().parents[2]
    sources = ['scripts/diagnostics/qualify_reconstruction.py','src/warpSPHCore/polyfit/polyfit.py',
               'src/warpSPHCore/polyfit/wp_polyfit.py']
    result = dict(device=args.device,precision=args.precision,torch=torch.__version__,warp=wp.__version__,
        formulation='frozen diffuse MLS/LABFM spatial derivative, RK4, periodic 1D, a=1, Voronoi quadrature',
        source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},rows=rows,rates=rates)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
