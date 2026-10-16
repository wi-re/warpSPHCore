#!/usr/bin/env python3
"""Separate frozen MFM face moments from h/volume iteration and evolution.

Manufactured Lagrangian sound-wave displacement at one quarter period;
unclosed and projected geometry remain distinct formulations.
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
    ap.add_argument('--n', type=int, nargs='+', default=[16, 32, 64])
    ap.add_argument('--jitter', type=float, nargs='+', default=[0., .3])
    ap.add_argument('--iterations', type=int, nargs='+', default=[0, 1, 3, 12, 48])
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    os.environ['warpSPHCore_PRECISION'] = 'float64'
    sys.path.insert(0, str(root/'higherOrderSPH/harness'))
    import torch
    import warp as wp
    from warpSPHCore import DomainDescription, ParticleState, KernelFunctions
    from warpSPHCore.mfm import MeshlessGeometry, mfmRates
    from warpSPHCore.mfm.geometry import _scatter
    from gizmo_gresho import lattice
    from mfm_driver import supports_from_volume

    wp.init()
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    result = dict(formulation=__doc__, precision='float64', device=a.device,
                  n=a.n, jitter=a.jitter, iterations=a.iterations, rows=[])
    result['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), root/'src/warpSPHCore/mfm/geometry.py',
                                        root/'src/warpSPHCore/mfm/scheme.py',
                                        root/'higherOrderSPH/harness/gizmo_gresho.py']}
    for jitter in a.jitter:
        for n in a.n:
            x0 = torch.tensor(lattice(n, jitter, 42), dtype=torch.float64, device=a.device)
            phase = 2*math.pi*x0[:, 0]
            # Integrate the linear wave's velocity at the initial material label.
            x = x0.clone()
            x[:, 0] += 1e-4/(2*math.pi)*(torch.cos(phase-math.pi/2)-torch.cos(phase))
            x = x.remainder(1)
            N = len(x)
            dom = DomainDescription(x.new_zeros(2), x.new_ones(2),
                                    torch.ones(2, dtype=torch.bool, device=a.device), 2)
            h0 = supports_from_volume(x.new_full((N,), 1/N), 28, 2)
            for iterations in a.iterations:
                p = ParticleState(positions=x, supports=h0.clone(), masses=x.new_ones(N),
                                  densities=x.new_ones(N),
                                  kinds=torch.zeros(N, dtype=torch.int32, device=a.device))
                for _ in range(iterations):
                    g = MeshlessGeometry.build(p, dom, KernelFunctions.Wendland2, closure='none')
                    p.supports = supports_from_volume(g.volume, 28, 2)
                for closure in ('none', 'project'):
                    g = MeshlessGeometry.build(p, dom, KernelFunctions.Wendland2, closure=closure)
                    frac = g.faceFraction()
                    tensor = g.A[:, :, None]*g.d[:, None, :]
                    moment = (_scatter(N, g.i, frac[:, None, None]*tensor)
                              + _scatter(N, g.j, (1-frac)[:, None, None]*tensor))/g.volume[:, None, None]
                    moment_error = (moment-torch.eye(2, dtype=x.dtype, device=a.device)).square().sum((1,2)).mean().sqrt()/math.sqrt(2)
                    leak = g.closure()
                    rates, _ = mfmRates(g, x.new_ones(N), x.new_zeros((N,2)), x.new_ones(N),
                                       1.4, order=1)
                    # For a static uniform MFM state, momentum rate is exactly -P*leak.
                    torch.testing.assert_close(rates[:, 1:3], -leak, rtol=1e-8, atol=1e-12)
                    row = dict(n=n, particles=N, jitter=jitter, support_iterations=iterations,
                        closure=closure,
                        support_fixed_point_residual=float((p.supports/supports_from_volume(g.volume,28,2)-1).abs().max()),
                        closure_norm_over_face_norm=float(leak.norm()/g.A.norm()),
                        first_moment_relative_rms=float(moment_error),
                        volume_sum_error=float(g.volume.sum()-1),
                        uniform_acceleration_rms=float((rates[:, 1:3]/g.volume[:, None]).square().sum(1).mean().sqrt()))
                    result['rows'].append(row)
                    print(json.dumps(row), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
