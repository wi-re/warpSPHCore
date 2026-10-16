#!/usr/bin/env python3
"""Serial frontend simulation A/Bs, including final-state and trajectory checks.

Run with the warp environment and both repositories' src directories on
PYTHONPATH. Each trial is a fresh process. No profiler or video in timed runs.
The 64-bit control uses CURRENT sources; it is not an old checkout.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = {
    'comp2d-eager': ('sod2d', 'CompSPH', 100, 'eager'),
    'comp2d-graph': ('sod2d', 'CompSPH', 100, 'graph'),
    'crk2d-eager': ('sod2d', 'CRKSPH', 100, 'eager'),
    'crk2d-graph': ('sod2d', 'CRKSPH', 100, 'graph'),
    'delta-eager': ('dambreak', 'sun2017DeltaSPH', 70, 'eager'),
    'delta-graph': ('dambreak', 'sun2017DeltaSPH', 70, 'graph'),
    'comp3d-eager': ('sod3d', 'CompSPH', 40, 'eager'),
    'crk2d-large': ('sod2d', 'CRKSPH', 1600, 'eager'),
    'comp3d-large': ('sod3d', 'CompSPH', 160, 'eager'),
}
POLICIES = {
    'wide-fused': ('int64', 'int64', 'fused', 'input', 'scheme'),
    'narrow-fused': ('int32', 'int32', 'fused', 'input', 'scheme'),
    'narrow-split': ('int32', 'int32', 'split', 'input', 'scheme'),
    'automatic': ('int32', 'int32', 'auto', 'auto', 'scheme'),
    'automatic-eager-grid': ('int32', 'int32', 'auto', 'auto', 'eager'),
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(args):
    ids, offsets, collection, order, grid_build = POLICIES[args.policy]
    os.environ.update(warpSPHCore_PRECISION='float32', warpSPHCore_INDEX_DTYPE=ids,
                      warpSPHCore_OFFSET_DTYPE=offsets, WARPSPHCORE_PROFILING='0')
    os.environ.pop('WARPSPHCORE_NEIGHBOR_LANES', None)
    import torch
    import warp as wp
    import warpSPH
    import warpSPHIntegrators
    from warpSPH.cases import importAll
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase
    from warpSPHCore.radiusSearch.options import resolveCollection, resolveQueryOrder

    torch.set_grad_enabled(False)
    torch.manual_seed(17)
    wp.init()
    importAll()
    case_name, scheme, nx, execution = SCENARIOS[args.scenario]
    base = getCase(case_name)
    stamps = {}
    search_calls = []
    next_step = -1
    # Observe actual low-level dispatch, not just the config values. This adds
    # host bookkeeping only on CSR rebuilds; no timing events or device reads.
    wrapper = importlib.import_module('warpSPHCore.radiusSearch.compactHash.wrapper')
    original_search = wrapper.radiusSearchOnCompactHashMap

    def observe(*a, **kw):
        query, reference = a[1:3]
        resolved = (resolveCollection(kw.get('collection', 'auto'), query.device),
                    resolveQueryOrder(kw.get('queryOrder', 'auto'), query, reference))
        adjacency = original_search(*a, **kw)
        search_calls.append(dict(step=next_step, collection=resolved[0], order=resolved[1],
                                 particles=query.shape[0], edges=adjacency.numEdges,
                                 capacity=adjacency.paddedJ.numel()))
        return adjacency

    wrapper.radiusSearchOnCompactHashMap = observe

    def configure(ctx):
        ctx.config.compute.neighborCollection = collection
        ctx.config.compute.neighborQueryOrder = order
        ctx.config.compute.neighborGridBuild = grid_build
        ctx.config.compute.validate()
        if base.configureScheme:
            base.configureScheme(ctx)

    def build_system(ctx):
        # SDF sampling differentiates the geometry to obtain boundary normals.
        # Keep this setup-only AD enabled; the simulation remains a value run.
        if case_name == 'dambreak':
            with torch.enable_grad():
                return base.buildSystem(ctx)
        return base.buildSystem(ctx)

    def diagnostics(ctx, state):
        nonlocal next_step
        values = base.diagnostics(ctx, state) if base.diagnostics else {}
        i = next_step
        if i == args.warmup - 1:
            torch.cuda.reset_peak_memory_stats()
        # The span between completed diagnostics includes integrator, next-dt
        # hook, diagnostics and loop bookkeeping. No new per-step sync here.
        stamps[i] = time.perf_counter()
        next_step += 1
        return values

    case = dataclasses.replace(base, buildSystem=build_system, configureScheme=configure, diagnostics=diagnostics)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = run(case, scheme=scheme, nx=nx, nSteps=args.warmup + args.active,
                 execution=execution, cudaGraph=None, pipelineOutputs=args.pipeline,
                 device='cuda', precision='float32', plot=False, video=False,
                 store=False, quiet=True, progress=True,
                 velocityAlarmPlotInterval=1, stallProgress=1e-3)
    torch.cuda.synchronize()
    full_wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    if result.diverged or result.stopReason is not None or result.nSteps != args.warmup + args.active:
        raise RuntimeError(f'run stopped: {result.nSteps}, {result.stopReason}')
    rows = [r for r in result.trajectory if r['step'] >= args.warmup]
    intervals = [r['stepTime_ms'] for r in rows]
    loop_intervals = [1000 * (stamps[i] - stamps[i - 1])
                      for i in range(args.warmup, args.warmup + args.active)]
    if len(rows) != args.active or not all(math.isfinite(x) and x > 0 for x in intervals + loop_intervals):
        raise RuntimeError('missing or invalid active timing samples')
    if not all(b['t'] > a['t'] for a, b in zip(result.trajectory, result.trajectory[1:])):
        raise RuntimeError('simulation time did not advance every step')
    tensors = {}
    for name, value in vars(result.state.state).items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f'nonfinite final state: {name}')
            tensors[name] = value.detach().cpu().clone()
    if 'positions' not in tensors:
        raise RuntimeError('no final particle positions')
    state_path = args.output.with_suffix('.pt')
    torch.save(tensors, state_path)
    trajectory = []
    for row in result.trajectory:
        scalar_row = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        for key, value in scalar_row.items():
            if not math.isfinite(value):
                # _statBlock intentionally reports NaN for an empty set of
                # no-penetration corrections. Preserve this as JSON null;
                # physical state and all populated diagnostics must be finite.
                if (key.startswith('nopenshiftMag') and math.isnan(value)
                        and scalar_row.get('nopenshiftNActive') == 0):
                    scalar_row[key] = None
                else:
                    raise RuntimeError(f'nonfinite populated diagnostic {key}: {value}')
        trajectory.append(scalar_row)
    graph = result.ctx.scratch.get('stepGraph')
    roots = [ROOT / 'src/warpSPHCore', Path(warpSPH.__file__).resolve().parent,
             Path(warpSPHIntegrators.__file__).resolve().parent]
    paths = [p for root in roots for p in sorted(root.rglob('*.py'))] + [Path(__file__)]
    report = dict(complete=True, scenario=args.scenario, policy=args.policy,
                  case=case_name, scheme=scheme, nx=nx, particles=tensors['positions'].shape[0],
                  execution_requested=execution, compute=dataclasses.asdict(result.ctx.config.compute),
                  index_dtype=ids, offset_dtype=offsets, warmup=args.warmup, active=args.active,
                  median_integrator_ms=statistics.median(intervals),
                  runner_interval_scope='wall throughput with overlapping outputs' if args.pipeline else 'CUDA-event integrator interval',
                  mean_integrator_ms=statistics.mean(intervals), integrator_ms=intervals,
                  mean_loop_ms=statistics.mean(loop_intervals), median_loop_ms=statistics.median(loop_intervals),
                  loop_ms=loop_intervals, full_run_including_setup_wall_s=full_wall,
                  active_peak_torch_allocated_bytes=peak, trajectory=trajectory,
                  undefined_diagnostic_scope='null nopenshiftMag statistics mean zero active no-penetration corrections',
                  final_state_finite=True, final_state_file=str(state_path),
                  final_state={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                                      sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest())
                               for k, v in tensors.items()},
                  search_dispatches=search_calls,
                  search_dispatch_scope='Python CSR builds, including setup and capture validation; no GPU replay counts. With pipelining, step labels follow diagnostic completion and may precede the true build step by one.',
                  graph=None if graph is None else {k: getattr(graph, k) for k in ('captures', 'replays', 'fallbacks', 'disabled')},
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, warp=wp.__version__,
                  command=sys.argv, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save(args.output, report)
    print(f'RESULT {args.scenario} {args.policy}: N={report["particles"]}, '
          f'integrator={report["mean_integrator_ms"]:.3f} ms, loop={report["mean_loop_ms"]:.3f} ms, '
          f'graph={report["graph"]}', flush=True)


def compare(reports):
    import torch
    comparisons = []
    for scenario in dict.fromkeys(r['scenario'] for r in reports):
        group = [r for r in reports if r['scenario'] == scenario]
        baseline = group[0]
        reference = torch.load(baseline['final_state_file'], weights_only=True)
        ref_rows = [{k: v for k, v in row.items() if k != 'stepTime_ms'} for row in baseline['trajectory']]
        for report in group[1:]:
            candidate = torch.load(report['final_state_file'], weights_only=True)
            fields = {}
            if candidate.keys() != reference.keys():
                raise RuntimeError(f'{scenario}: final tensor field set changed')
            for key, a in reference.items():
                b = candidate[key]
                equal = torch.equal(a, b)
                fields[key] = dict(bitwise_equal=equal, max_abs_difference=float((a.double() - b.double()).abs().max()) if a.numel() else 0.)
            rows = [{k: v for k, v in row.items() if k != 'stepTime_ms'} for row in report['trajectory']]
            comparisons.append(dict(scenario=scenario, reference=baseline['command'], candidate=report['command'],
                                    all_final_tensors_bitwise_equal=all(f['bitwise_equal'] for f in fields.values()),
                                    trajectory_scalars_equal=ref_rows == rows, fields=fields))
    return comparisons


def suite(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports, failures = [], []
    for scenario in args.scenarios:
        for repeat in range(args.repeats):
            # Rotate order to reduce first/last ordering bias, all runs serial.
            policies = args.policies[repeat % len(args.policies):] + args.policies[:repeat % len(args.policies)]
            for policy in policies:
                output = args.output_dir / f'{scenario}-{policy}-r{repeat}.json'
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--scenario', scenario,
                           '--policy', policy, '--warmup', str(args.warmup), '--active', str(args.active), '--output', str(output)]
                if args.pipeline:
                    command.append('--pipeline')
                print(f'START {scenario} {policy} repeat={repeat}', flush=True)
                with output.with_suffix('.log').open('w') as log:
                    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if done.returncode:
                    failures.append(dict(scenario=scenario, policy=policy, repeat=repeat, log=str(output.with_suffix('.log'))))
                    print(f'FAILED {failures[-1]}', flush=True)
                else:
                    report = json.loads(output.read_text())
                    reports.append(report)
                    print(f'DONE loop={report["mean_loop_ms"]:.3f} ms, integrator={report["mean_integrator_ms"]:.3f} ms', flush=True)
                save(args.output_dir / 'progress.json', dict(completed=len(reports), failures=failures))
    comparisons = compare(reports)
    aggregated = []
    for scenario in args.scenarios:
        for policy in args.policies:
            group = [r for r in reports if r['scenario'] == scenario and r['policy'] == policy]
            if group:
                aggregated.append(dict(scenario=scenario, policy=policy, particles=group[0]['particles'],
                                       trials=len(group), loop_ms=[r['mean_loop_ms'] for r in group],
                                       integrator_ms=[r['mean_integrator_ms'] for r in group],
                                       median_trial_loop_ms=statistics.median(r['mean_loop_ms'] for r in group),
                                       median_trial_integrator_ms=statistics.median(r['mean_integrator_ms'] for r in group)))
    save(args.output_dir / 'summary.json', dict(complete=not failures, failures=failures, aggregates=aggregated,
                                              comparisons=comparisons, reports=[str(p) for p in sorted(args.output_dir.glob('*-r*.json'))],
                                              timing_scope='Integrator: runner CUDA-event interval including host-fed GPU idle gaps. Loop: completed-diagnostics wall intervals, includes timestep and diagnostics, pipeline disabled. Compilation/setup excluded by warmup. No profiler/video.',
                                              baseline_scope='All trials use current sources; wide-fused restores 64-bit storage and old search policy, not all historical implementation details.'))
    if failures:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--pipeline', action='store_true', help='exercise the frontend default overlapped output loop')
    ap.add_argument('--scenario', choices=SCENARIOS)
    ap.add_argument('--policy', choices=POLICIES)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    ap.add_argument('--policies', nargs='+', choices=POLICIES, default=['wide-fused', 'narrow-fused', 'automatic'])
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'docs/regression/frontend_search_2026_10_15')
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=40)
    ap.add_argument('--active', type=int, default=160)
    args = ap.parse_args()
    if args.warmup < 1 or args.active < 1:
        ap.error('warmup and active must be positive')
    worker(args) if args.worker else suite(args)


if __name__ == '__main__':
    main()
