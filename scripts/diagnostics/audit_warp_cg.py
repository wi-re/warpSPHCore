#!/usr/bin/env python3
"""Record upstream CG numerical behavior without treating its residual as a certificate."""

import argparse
import json
from pathlib import Path
import sys

import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from _warp_cg_reference import WarpCGReference
from warpSPHCore.linalg import cg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is required')
    wp.init()
    rows = []
    for dtype in (torch.float32, torch.float64):
        b = torch.ones(4, device=args.device, dtype=dtype)
        cases = [
            ('identity', lambda v: v, b, None, 0., 8),
            ('zero_warm', lambda v: v, b * 0, b, 0., 8),
            ('negative_identity', lambda v: -v, b, None, 1e-5, 8),
            ('zero_operator', lambda v: v * 0, b, None, 1e-5, 8),
            ('nonfinite', lambda v: v * float('nan'), b, None, 1e-5, 8),
            ('tiny_rhs_zero_tolerance', lambda v: v * 2, b * 1e-19, None, 0., 8),
            ('zero_budget', lambda v: v * 2, b, None, 1e-5, 0),
        ]
        for name, action, rhs, x0, rtol, budget in cases:
            reference = cg(action, rhs, x0, maxiter=budget, rtol=rtol)
            upstream = WarpCGReference(action, rhs, maxiter=budget, rtol=rtol)
            observation = upstream.observe(rhs, x0)
            rows.append(dict(case=name, dtype=str(dtype), upstream_scheduled=observation.scheduled,
                             upstream_recurrence=observation.recurrence, true_residual=observation.trueResidual,
                             target=observation.target, finite=observation.finite,
                             independently_verified=observation.verified, core_status=reference.statusInt(),
                             core_finite=bool(reference.x.isfinite().all()), core_updates=int(reference.iterations)))
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(80, 80, dtype=torch.float64, generator=g))
    A = ((Q * torch.logspace(0, 5, 80, dtype=torch.float64)) @ Q.T).float().to(args.device)
    b = torch.randn(80, generator=g).to(args.device)
    upstream = WarpCGReference(lambda v: A @ v, b, maxiter=2000, rtol=1e-5)
    observed = upstream.observe(b)
    rows.append(dict(case='ill_conditioned_float32', dtype=str(b.dtype), upstream_scheduled=observed.scheduled,
                     upstream_recurrence=observed.recurrence, true_residual=observed.trueResidual,
                     target=observed.target, independently_verified=observed.verified, finite=observed.finite))
    report = dict(torch=torch.__version__, warp=wp.__version__, device=args.device,
                  gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None, cases=rows)
    # JSON cannot encode NaN/Infinity portably. Preserve the observation as text.
    import math
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = str(value)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
