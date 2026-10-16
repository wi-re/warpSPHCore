#!/usr/bin/env python3
"""Capability probes for production MFM/MFV; raw invariants, no timing gates.

Uses the existing reference stepper, not a second hydrodynamics frontend.
Run with PYTHONPATH=src; each precision must use a fresh Python process.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', choices=['wave', 'gresho', 'sod'], required=True)
    ap.add_argument('--n', type=int, nargs='+', required=True,
                    help='Particle count in 1D, particles per axis in 2D.')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--precision', choices=['float32', 'float64'], default='float64')
    ap.add_argument('--backend', choices=['torch', 'warp'], default='warp')
    ap.add_argument('--modes', nargs='+', choices=['MFM', 'MFV'], default=['MFM', 'MFV'])
    ap.add_argument('--closures', nargs='+', choices=['none', 'project'], default=['project'])
    ap.add_argument('--dim', type=int, choices=[1, 2], default=1)
    ap.add_argument('--jitter', type=float, default=0.)
    ap.add_argument('--repeat-fine', type=int, default=1)
    ap.add_argument('--h-iters', type=int, default=0)
    ap.add_argument('--cg-tol', type=float)
    ap.add_argument('--tend', type=float)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    os.environ['warpSPHCore_PRECISION'] = args.precision
    sys.path.insert(0, str(root / 'higherOrderSPH/harness'))
    import numpy as np
    import torch
    import warp as wp
    from warpSPHCore import DomainDescription
    from warpSPHCore.enumTypes import KernelFunctions
    import mfm_driver as drv
    import exact_riemann
    from gizmo_gresho import lattice, metrics
    from run_mfm_gresho import profile, vortex_velocity

    wp.init()
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    dtype = getattr(torch, args.precision)
    drv.BACKEND, drv.CG_TOL, drv.H_ITERS = args.backend, args.cg_tol, args.h_iters
    sources = sorted((root / 'src/warpSPHCore/mfm').glob('*.py'))
    sources += [Path(__file__), root / 'higherOrderSPH/harness/mfm_driver.py',
                root / 'higherOrderSPH/harness/gizmo_gresho.py',
                root / 'higherOrderSPH/harness/run_mfm_gresho.py',
                root / 'higherOrderSPH/harness/exact_riemann.py']
    result = dict(configuration=vars(args) | {'output': str(args.output)},
                  runtime=dict(torch=torch.__version__, warp=wp.__version__),
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources}, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    def geometry(st, dom, kernel, closure):
        return drv.geometry(st, dom, kernel, closure)

    def run(n, mode, closure, repeat):
        dim = 2 if args.suite == 'gresho' else (1 if args.suite == 'sod' else args.dim)
        gamma = 5/3 if args.suite == 'gresho' else 1.4
        kernel = KernelFunctions.Wendland4 if args.suite == 'gresho' else KernelFunctions.Wendland2
        nngb = 30. if args.suite == 'gresho' else (7. if dim == 1 else 28.)
        if dim == 1:
            k = torch.arange(n, dtype=dtype, device=args.device)
            x = ((k+.5+args.jitter*.5*torch.sin(7.3*k))/n)[:, None]
        else:
            x = torch.tensor(lattice(n, args.jitter, 42), dtype=dtype, device=args.device)
        N = len(x)
        dom = DomainDescription(torch.zeros(dim, dtype=dtype, device=args.device),
                                torch.ones(dim, dtype=dtype, device=args.device),
                                torch.ones(dim, dtype=torch.bool, device=args.device), dim)
        h = drv.supports_from_volume(torch.full((N,), 1/N, dtype=dtype, device=args.device), nngb, dim)
        dummy = drv.MFMState(x, torch.ones(N, dtype=dtype, device=args.device),
                              torch.zeros(N, dim+2, dtype=dtype, device=args.device), h)
        # Retain the historical four support iterations, then rebuild at the
        # actual final h before assigning masses; do not hide an initial rho bias.
        for _ in range(4):
            g = geometry(dummy, dom, kernel, closure)
            dummy.h = drv.supports_from_volume(g.volume, nngb, dim)
        g = geometry(dummy, dom, kernel, closure)
        if args.suite == 'gresho':
            vel, radius = vortex_velocity(x, x.new_full((2,), .5), x.new_ones(2))
            _, pressure = profile(radius)
            rho = x.new_ones(N)
        elif args.suite == 'wave':
            phase = torch.sin(2*math.pi*x[:, 0])
            rho = 1 + 1e-4*phase
            vel = x.new_zeros((N, dim)); vel[:, 0] = 1e-4*math.sqrt(gamma)*phase
            pressure = 1 + gamma*1e-4*phase
        else:
            high = (x[:, 0] >= .25) & (x[:, 0] < .75)
            rho = torch.where(high, 1., .125).to(dtype)
            pressure = torch.where(high, 1., .1).to(dtype)
            vel = x.new_zeros((N, 1))
        st = drv.init_state(x, rho*g.volume, rho, vel, pressure, dummy.h, gamma)
        initial = st.Q.sum(0).clone()
        scale = st.Q.abs().sum(0).clamp_min(1.)
        initial_density_error = float((st.Q[:, 0]/g.volume-rho).abs().max())
        initial_support_residual = float((dummy.h/drv.supports_from_volume(g.volume, nngb, dim)-1).abs().max())
        end = args.tend if args.tend is not None else (
            3. if args.suite == 'gresho' else (.12 if args.suite == 'sod' else 1/math.sqrt(gamma)))
        ke0 = float((st.Q[:, 1:-1].square().sum(1)/(2*st.Q[:, 0])).sum())
        kemax, kemin, rebound = 1., 1., 0.
        minmass, mininternal = float(st.Q[:, 0].min()), math.inf
        maxdrift = torch.zeros_like(initial)
        steps = 0
        print(f'START {args.suite} {args.backend} {mode} {closure} n={n} repeat={repeat}', flush=True)
        while st.t < end - 1e-12:
            st, g, dt = drv.step(st, dom, kernel, gamma, nngb, mode=mode, closure=closure,
                                  order=2, tmax=end)
            if not bool(torch.isfinite(st.Q).all() and torch.isfinite(st.pos).all()):
                raise AssertionError('Nonfinite evolution state')
            mass = st.Q[:, 0]
            internal = st.Q[:, -1] - st.Q[:, 1:-1].square().sum(1)/(2*mass)
            minmass = min(minmass, float(mass.min()))
            mininternal = min(mininternal, float(internal.min()))
            if minmass <= 0 or mininternal <= 0:
                raise AssertionError(f'Nonpositive raw mass/internal energy: {minmass}, {mininternal}')
            maxdrift = torch.maximum(maxdrift, (st.Q.sum(0)-initial).abs())
            kinetic = float((st.Q[:, 1:-1].square().sum(1)/(2*mass)).sum())
            ke = kinetic/ke0 if ke0 > 0 else None
            if ke is not None:
                kemax = max(kemax, ke)
                kemin = min(kemin, ke)
                rebound = max(rebound, ke-kemin)
            steps += 1
            if steps % 400 == 0:
                print(f'  step {steps} t={st.t:.6f}/{end:.6f}', flush=True)
        g = geometry(st, dom, kernel, closure)
        r, v, p = drv.primitives(st.Q, g.volume, gamma)
        row = dict(n=n, particles=N, dim=dim, mode=mode, closure=closure, repeat=repeat,
                   jitter=args.jitter, kernel=kernel.name, nngb=nngb, steps=steps, time=st.t,
                   initial_density_error=initial_density_error,
                   initial_support_fixed_point_residual=initial_support_residual,
                   min_mass=minmass, min_internal_energy=mininternal,
                   final_density_min=float(r.min()), final_pressure_min=float(p.min()),
                   max_conservation_drift=maxdrift.tolist(),
                   scaled_conservation_drift=(maxdrift/scale).tolist(),
                   initial_totals=initial.tolist(), final_totals=st.Q.sum(0).tolist(),
                   ke_final=ke, ke_peak=kemax if ke0 > 0 else None,
                   ke_rebound=rebound if ke0 > 0 else None, kinetic_energy_final=kinetic,
                   deficient=int(g.deficient.sum()), condition_max=float(g.cond.max()))
        if args.suite == 'wave':
            phase = torch.sin(2*math.pi*(st.pos[:, 0]-math.sqrt(gamma)*end))
            dr = r-1-1e-4*phase
            dv = v[:, 0]-1e-4*math.sqrt(gamma)*phase
            amp = 1e-4*phase.square().mean().sqrt()
            row.update(density_relative_l2=float(dr.square().mean().sqrt()/amp),
                       density_demeaned_relative_l2=float((dr-dr.mean()).square().mean().sqrt()/amp),
                       density_mean_error=float(dr.mean()),
                       velocity_relative_l2=float(dv.square().mean().sqrt()/(math.sqrt(gamma)*amp)))
        elif args.suite == 'gresho':
            l1g, l1v, _ = metrics(st.pos.cpu().numpy(), v.cpu().numpy(), g.volume.cpu().numpy())
            row.update(l1_gizmo=l1g, l1_volume=l1v)
        else:
            xx = st.pos[:, 0].cpu().numpy()
            xr = np.where(xx >= .5, xx, 1-xx)
            exact = exact_riemann.sample((xr-.75)/end, 1., 0., 1., .125, 0., .1, gamma)
            exact[:, 1] *= np.where(xx >= .5, 1., -1.)
            rn, vn, pn = r.cpu().numpy(), v[:, 0].cpu().numpy(), p.cpu().numpy()
            plateau = (xx > .77) & (xx < .83)
            sort = np.argsort(xx)
            row.update(density_l1=float(np.abs(rn-exact[:, 0]).mean()),
                       velocity_l1=float(np.abs(vn-exact[:, 1]).mean()),
                       pressure_l1=float(np.abs(pn-exact[:, 2]).mean()),
                       velocity_tv=float(np.abs(np.diff(vn[sort])).sum()),
                       exact_velocity_tv=float(np.abs(np.diff(exact[sort, 1])).sum()),
                       plateau_samples=int(plateau.sum()),
                       plateau_velocity_std=float(vn[plateau].std()),
                       plateau_velocity_max_error=float(np.abs(vn[plateau]-exact[plateau, 1]).max()))
        row['final_state_sha256'] = hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes()
                                                              for t in (st.pos, st.Q, st.h))).hexdigest()
        assert max(row['scaled_conservation_drift']) < (2e-5 if dtype == torch.float32 else 1e-10)
        if closure == 'project' and args.suite == 'wave' and args.jitter == 0 and dim == 1:
            assert row['density_relative_l2'] < .25
        print(json.dumps(row, allow_nan=False), flush=True)
        return row

    for closure in args.closures:
        for mode in args.modes:
            for n in args.n:
                for repeat in range(args.repeat_fine if n == args.n[-1] else 1):
                    result['rows'].append(run(n, mode, closure, repeat))
                    checkpoint()
    result['observed_orders'] = []
    if args.suite == 'wave':
        for closure in args.closures:
            for mode in args.modes:
                rows = [r for r in result['rows'] if r['closure']==closure and r['mode']==mode and r['repeat']==0]
                for a, b in zip(rows, rows[1:]):
                    result['observed_orders'].append(dict(closure=closure, mode=mode, coarse=a['n'], fine=b['n'],
                        density=math.log(a['density_relative_l2']/b['density_relative_l2'])/math.log(b['n']/a['n']),
                        velocity=math.log(a['velocity_relative_l2']/b['velocity_relative_l2'])/math.log(b['n']/a['n'])))
    checkpoint()


if __name__ == '__main__':
    main()
