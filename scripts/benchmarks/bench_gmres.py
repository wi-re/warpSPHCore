#!/usr/bin/env python3
"""Verified GMRES on a nonsymmetric, fixed-CSR SPH action.

Report whole-call timing, observed pair work, optional promoted accumulation,
and peak Torch allocation. Search setup and compilation are excluded. This is
a frozen operator qualification, not a simulation or a whole-scheme speedup.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import warp as wp

from _work_metrics import WorkLedger
from warpSPHCore import (
    KernelFunctions, LaplacianScheme, GradientScheme, OperationProperties,
    ParticleState, SupportScheme, WarpOperation, radiusSearchCompactHashMap,
    warpOperation,
)
from warpSPHCore.linalg import gmres
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.util import generateNeighborTestData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--sides', type=int, nargs='+', default=[32, 256, 1024])
    parser.add_argument('--neighbors', type=int, nargs='+', default=[25, 55])
    parser.add_argument('--preconditioner-sides', nargs='+', choices=['left', 'right'], default=['right'])
    parser.add_argument('--restart', type=int, default=20)
    parser.add_argument('--maxiter', type=int, default=160)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--promote', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.sides, *args.neighbors, args.restart, args.maxiter, args.repeat, args.warmup) <= 0:
        parser.error('sizes, counts and budgets must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA device unavailable')
    torch.set_num_threads(2)
    wp.init()
    dtype = get_torch_precision()
    rtol = 2e-5 if dtype == torch.float32 else 1e-10
    rows = []
    for side in args.sides:
        for requested in args.neighbors:
            positions, supports, _, domain, dx = generateNeighborTestData(side, requested, 2, True, args.device)
            n = positions.shape[0]
            particles = ParticleState(
                positions=positions, supports=supports,
                masses=torch.full((n,), dx ** 2, dtype=dtype, device=args.device),
                densities=torch.ones(n, dtype=dtype, device=args.device),
                kinds=torch.zeros(n, dtype=torch.int32, device=args.device))
            adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
            edges = int(adjacency.numNeighbors.sum())
            props = OperationProperties(
                kernel=KernelFunctions.Wendland2, operation=WarpOperation.Laplacian,
                gradientMode=GradientScheme.Difference, laplacianMode=LaplacianScheme.Brookshaw,
                supportMode=SupportScheme.Gather)
            # Row-varying coefficients make the Laplacian action nonsymmetric.
            scale = dx ** 2 * (.5 + .25 * torch.sin(positions[:, 0] * 9))
            inverse = 1 / (1 + scale / dx ** 2)

            def action(v):
                lap = warpOperation(particles, props, domain, adjacency=adjacency,
                                    queryValues=v, referenceValues=v)
                return v - scale * lap

            generator = torch.Generator(device=args.device).manual_seed(17)
            known = torch.randn(n, dtype=dtype, device=args.device, generator=generator)
            b = action(known)
            for preconditionerSide in args.preconditioner_sides:
                kw = dict(rtol=rtol, maxiter=args.maxiter, restart=args.restart,
                          precond=inverse, preconditionerSide=preconditionerSide,
                          accumulationDtype=torch.float64 if args.promote else dtype,
                          history=False)
                for _ in range(args.warmup):
                    warm = gmres(action, b, **kw)
                    if warm.statusInt() < 0:
                        raise RuntimeError(f'warmup did not converge: {warm.statusInt()}')
                del warm
                if args.device == 'cuda':
                    torch.cuda.current_stream().synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                else:
                    baseline = None
                samples, intervals, diagnostics = [], [], []
                for _ in range(args.repeat):
                    if args.device == 'cuda':
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                    before = time.perf_counter()
                    result = gmres(action, b, **kw)
                    if args.device == 'cuda':
                        end.record()
                        end.synchronize()
                        intervals.append(start.elapsed_time(end))
                    samples.append((time.perf_counter() - before) * 1e3)
                    actual = (b - action(result.x)).norm()
                    if result.statusInt() < 0 or not bool(actual <= rtol * b.norm()):
                        raise RuntimeError('solve failed independent original-residual verification')
                    torch.testing.assert_close(result.trueResidual, actual, rtol=0, atol=0)
                    diagnostics.append(dict(status=result.statusInt(), iterations=int(result.iterations),
                                            loops=result.loopIterations, matvecs=result.matvecs,
                                            preconditionerApplications=result.preconditionerApplications,
                                            verifications=result.verifications, restarts=result.restarts,
                                            relativeResidual=float(actual / b.norm()),
                                            relativeSolutionError=float((result.x - known).norm() / known.norm())))
                    del result
                if any(row != diagnostics[0] for row in diagnostics):
                    raise RuntimeError('numerical work changed across identical samples')
                median = statistics.median(samples)
                counts = diagnostics[0]
                ledger = WorkLedger()
                for role, passes in [('Arnoldi_including_frozen', counts['matvecs'] - counts['verifications']),
                                     ('original_residual_verification', counts['verifications'])]:
                    ledger.pair('Brookshaw Laplacian', evaluated_nonself_pairs=edges - n,
                                candidate_entries=edges, passes=passes, role=role,
                                basis='actual periodic equal-support CSR; includes all scheduled callbacks')
                ledger.particle('identity_minus_row_scaled_laplacian', particles=n,
                                passes=counts['matvecs'])
                row = dict(n=n, requestedNeighbors=requested, edges=edges, meanDegree=edges / n,
                           preconditionerSide=preconditionerSide, dtype=str(dtype),
                           accumulationDtype=str(kw['accumulationDtype']), restart=args.restart,
                           wallMs=samples, medianWallMs=median, deviceIntervalsMs=intervals,
                           peakTorchBytes=None if baseline is None else torch.cuda.max_memory_allocated(),
                           incrementalPeakTorchBytes=None if baseline is None else torch.cuda.max_memory_allocated() - baseline,
                           **counts, work=ledger.report(wall_ms=median, timing_scope='whole warmed verified GMRES call'))
                rows.append(row)
                print(json.dumps({key: row[key] for key in ('n', 'meanDegree', 'preconditionerSide', 'medianWallMs',
                                                            'iterations', 'loops', 'relativeResidual')}), flush=True)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), root / 'src/warpSPHCore/linalg/krylov/gmres.py',
               root / 'src/warpSPHCore/linalg/operator.py', root / 'src/warpSPHCore/linalg/krylov/common.py',
               root / 'src/warpSPHCore/operations.py', root / 'src/warpSPHCore/coreOperations/wp_laplacian.py',
               root / 'src/warpSPHCore/autograd/lanes.py', root / 'src/warpSPHCore/type_config.py',
               root / 'scripts/benchmarks/_work_metrics.py']
    payload = dict(schemaVersion=1, scope=__doc__, torch=torch.__version__, warp=wp.__version__,
                   device=args.device, gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                   processPrecision=os.environ.get('warpSPHCore_PRECISION', 'float32'),
                   arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   sourceSha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                   records=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
