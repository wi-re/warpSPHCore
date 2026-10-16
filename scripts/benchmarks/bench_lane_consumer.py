#!/usr/bin/env python3
"""Run an installed warpSPH consumer with only curl/Laplacian lane defaults changed.

Other operators keep their existing defaults. No sibling source is edited.
The runner's step times are CUDA intervals, not a claim about complete loop
wall time; the separate full-run wall time includes setup/warmup/diagnostics.
"""

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='waveEquation')
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--trial-lanes', type=int, choices=[1, 2, 4, 8, 16, 32], default=1)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--active', type=int, default=20)
    parser.add_argument('--spec', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state-output', type=Path)
    args = parser.parse_args()
    if min(args.nx, args.warmup, args.active) <= 0:
        parser.error('sizes and step counts must be positive')
    # Benchmark comparison changes two per-operator defaults only. Global lane
    # overrides would also change density/gradient and confound attribution.
    os.environ.pop('WARPSPHCORE_NEIGHBOR_LANES', None)
    os.environ['WARPSPHCORE_PROFILING'] = '0'
    os.environ['warpSPHCore_PRECISION'] = 'float32'
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    import warp as wp
    import warpSPH
    from warpSPHCore import setNeighborLanes
    from bench_real_workload import parseAssignments, runProfiled

    if not torch.cuda.is_available():
        parser.error('requires an accessible CUDA device')
    torch.set_num_threads(2)
    setNeighborLanes(None)
    dispatches = {}
    for name, attribute in [('curl', '_CURL_SPEC'), ('laplacian', '_LAPLACIAN_SPEC')]:
        module = importlib.import_module(f'warpSPHCore.coreOperations.wp_{name}')
        setattr(module, attribute, replace(getattr(module, attribute), defaultNeighborLanes=args.trial_lanes))
        original = module.launchOperator

        def observe(*a, _original=original, _name=name, **kw):
            dispatches[_name] = dispatches.get(_name, 0) + 1
            return _original(*a, **kw)

        module.launchOperator = observe
    wp.init()
    torch.cuda.synchronize()
    before = time.perf_counter()
    result = runProfiled(args.nx, args.warmup, args.active, prof=None, forcing=False,
                         caseName=args.case, preset='none',
                         spec={'pipelineOutputs': False} | parseAssignments(args.spec))
    torch.cuda.synchronize()
    fullWall = time.perf_counter() - before
    if result.diverged or result.stopReason is not None or result.nSteps != args.warmup + args.active:
        raise RuntimeError(f'consumer stopped: {result.nSteps} steps, {result.stopReason}')
    rows = [row for row in result.trajectory if row.get('step', -1) >= args.warmup]
    times = [row['stepTime_ms'] for row in rows]
    if len(times) != args.active or not all(math.isfinite(value) and value > 0 for value in times):
        raise RuntimeError('consumer did not return the expected finite timed steps')
    state = result.state.state
    tensors = {}
    for name in ('positions', 'u', 'v', 'velocities', 'densities', 'masses', 'internalEnergies'):
        value = getattr(state, name, None)
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f'nonfinite final consumer state: {name}')
            tensors[name] = value.detach().cpu().clone()
    if args.state_output:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, args.state_output)
    frontendRoot = Path(warpSPH.__file__).resolve().parent
    paths = sorted((ROOT / 'src/warpSPHCore').rglob('*.py')) + sorted(frontendRoot.rglob('*.py'))
    paths += [Path(__file__), Path(__file__).with_name('bench_real_workload.py')]
    diagnostics = [{k: float(v) for k, v in row.items()
                    if isinstance(v, (int, float)) and math.isfinite(v)} for row in result.trajectory]
    report = dict(case=args.case, nx=args.nx, particles=state.positions.shape[0], trial_lanes=args.trial_lanes,
                  warmup=args.warmup, active=args.active, median_step_device_interval_ms=statistics.median(times),
                  step_device_interval_ms=times, full_run_wall_seconds=fullWall, complete=True,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, warp=wp.__version__,
                  frontend=str(frontendRoot), trajectory=diagnostics, numerical_state_finite=True,
                  final_state={name: dict(shape=list(value.shape), norm=float(value.double().norm())) for name, value in tensors.items()},
                  python_operator_dispatches_including_recording=dispatches,
                  dispatch_count_scope='Python calls include setup/recording; not GPU replay or interaction counts',
                  timing_scope=__doc__, command=sys.argv,
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'{args.case}, N={state.positions.shape[0]}, trial lanes={args.trial_lanes}: '
          f'{statistics.median(times):.4f} ms/step; {dispatches}')


if __name__ == '__main__':
    main()
