"""Check ideal-gas MFM/MFV rates and a short periodic update with fixed supports.

This bounded first-order-in-time illustration owns its drift/rebuild policy.
For the qualified MUSCL-Hancock evolution/support policy, run the existing
scripts/diagnostics/qualify_mfm.py harness instead.
"""

import argparse
import json

import torch
import warp as wp

from warpSPHCore import DomainDescription, KernelFunctions, ParticleState, get_torch_precision
from warpSPHCore.mfm import MeshlessGeometry, MeshlessWarp, conserved, mfmRates, primitives


@torch.no_grad()
def run(*, device='cpu', mode='MFM', backend='warp', steps=4):
    wp.init()
    n, gamma, dt = 32, 1.4, 1e-4
    x = (torch.arange(n, dtype=get_torch_precision(), device=device)+.5)/n
    positions, h = x[:, None], torch.full_like(x, 3.5/n)
    kinds = torch.zeros(n, dtype=torch.int32, device=device)
    domain = DomainDescription(x.new_tensor([0.]), x.new_tensor([1.]),
                               torch.ones(1, dtype=torch.bool, device=device), 1)

    def geometry():
        if backend == 'warp':
            return MeshlessWarp.build(positions, h, domain, KernelFunctions.Wendland2, closure='project')
        state = ParticleState(positions, h, torch.ones_like(h), kinds)
        return MeshlessGeometry.build(state, domain, KernelFunctions.Wendland2, closure='project')

    g = geometry()
    rho = 1+.01*(2*torch.pi*x).sin()
    velocity = (.01*(2*torch.pi*x).sin())[:, None]
    pressure = 1+.014*(2*torch.pi*x).sin()
    Q = conserved(rho*g.volume, rho, velocity, pressure, gamma)
    initial_mass, initial_total = Q[:, 0].clone(), Q.sum(0)
    max_drift = 0.
    for _ in range(steps):
        rho, velocity, pressure = primitives(Q, g.volume, gamma)
        # dt=0 disables the half-step predictor for this explicit Euler example.
        if backend == 'warp':
            rate = g.rates(rho, velocity, pressure, gamma, mode=mode, order=2, dt=0.)
        else:
            rate, _ = mfmRates(g, rho, velocity, pressure, gamma, mode=mode, order=2, dt=0.)
        relative_leak = rate.sum(0).abs()/rate.abs().sum(0).clamp_min(1)
        tol = 5e-6 if x.dtype == torch.float32 else 2e-12
        if not bool((relative_leak < tol).all()):
            raise AssertionError('rates violate global conservation')
        if mode == 'MFM' and bool(rate[:, 0].count_nonzero()):
            raise AssertionError('MFM must have exactly zero particle mass rate')
        Q = Q+dt*rate
        mass = Q[:, 0]
        raw_internal = Q[:, -1]/mass-.5*(Q[:, 1:-1]/mass[:, None]).square().sum(-1)
        if not bool(torch.isfinite(Q).all() & (mass > 0).all() & (raw_internal > 0).all()):
            raise AssertionError('invalid unfloored conserved state')
        max_drift = max(max_drift, float(((Q.sum(0)-initial_total).abs()/initial_total.abs().clamp_min(1)).max()))
        positions = (positions+dt*velocity).remainder(1.).contiguous()
        g = geometry()  # Changed positions invalidate the old geometry/CSR.
    if max_drift > tol:
        raise AssertionError('updated state violates global conservation')
    if mode == 'MFM':
        torch.testing.assert_close(Q[:, 0], initial_mass, rtol=0, atol=0)
    return dict(device=str(x.device), dtype=str(x.dtype), backend=backend, mode=mode,
                steps=steps, time=steps*dt, max_conservation_drift=max_drift,
                min_mass=float(mass.min()), min_raw_internal_energy=float(raw_internal.min()),
                max_particle_mass_change=float((mass-initial_mass).abs().max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--mode', choices=['MFM', 'MFV'], default='MFM')
    parser.add_argument('--backend', choices=['torch', 'warp'], default='warp')
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    print(json.dumps(run(device=args.device, mode=args.mode, backend=args.backend, steps=args.steps), indent=2))
