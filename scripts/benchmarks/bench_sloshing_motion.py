#!/usr/bin/env python3
"""Matched current-code sloshing windows initialized from recorded TC10 frames.

Integrated fields and static particle/ghost metadata are restored, derived
fields and adjacency are rebuilt. This is a workload fixture, not a claim of
bitwise continuation of the historical trajectory. Headline runs have no
timing instrumentation; --measure-search is a separate diagnostic pass.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import bench_frontend_search as benchmark

ROOT = benchmark.ROOT
DEFAULT_FIXTURES = ROOT / 'docs/regression/sloshing_motion_2026_10_15/fixtures'


def worker(args):
    ids, offsets, *_ = benchmark.POLICIES[args.policy]
    os.environ.update(warpSPHCore_PRECISION='float32', warpSPHCore_INDEX_DTYPE=ids,
                      warpSPHCore_OFFSET_DTYPE=offsets, WARPSPHCORE_PROFILING='0')
    import numpy as np
    import torch
    import warp as wp
    import warpSPH.runner as runner
    fixture_path = args.fixtures / f'{args.phase}.npz'
    manifest = json.loads((args.fixtures / 'manifest.json').read_text())
    arrays = dict(np.load(fixture_path, allow_pickle=False))
    fixture_time = float(arrays.pop('time'))
    setup_checks = {}
    observations = []
    physical_step = -1
    run_result = None

    def instrument(function, section):
        def invoke(*a, **kw):
            start_step = physical_step
            measured = args.measure_search and not torch.cuda.is_current_stream_capturing()
            if measured:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                wall_start = time.perf_counter()
            out = function(*a, **kw)
            item = dict(step=start_step, section=section)
            if measured:
                end.record()
                item.update(events=(begin, end), host_call_ms=(time.perf_counter() - wall_start) * 1000)
            if section == 'search':
                item.update(edges=out.numEdges, particles=out.numRows)
            observations.append(item)
            return out
        return invoke

    wrapper = importlib.import_module('warpSPHCore.radiusSearch.compactHash.wrapper')
    wrapper.radiusSearchOnCompactHashMap = instrument(wrapper.radiusSearchOnCompactHashMap, 'search')
    verlet = importlib.import_module('warpSPHCore.radiusSearch.verlet.build')
    verlet.radiusSearchCompactHashMap_ = instrument(verlet.radiusSearchCompactHashMap_, 'grid_and_search')
    original_run = runner.run

    def run(case, **kw):
        nonlocal physical_step, run_result
        original_build, original_ic, original_post = case.buildSystem, case.initialConditions, case.postStep

        def build(ctx):
            with torch.enable_grad():
                return original_build(ctx)

        def initial(ctx, system):
            nonlocal physical_step
            if original_ic:
                original_ic(ctx, system)
            state = system.state
            loaded = {}
            for key, value in arrays.items():
                old = getattr(state, key)
                tensor = torch.from_numpy(value.copy()).to(device=ctx.device, dtype=old.dtype)
                if tensor.shape != old.shape:
                    raise RuntimeError(f'fixture geometry shape mismatch for {key}: {tensor.shape} vs {old.shape}')
                if key not in ('positions', 'velocities', 'densities'):
                    exact = torch.equal(tensor, old)
                    close = exact or (old.is_floating_point() and bool(torch.allclose(tensor, old, atol=1e-8, rtol=1e-6)))
                    setup_checks[key] = dict(exact=exact, compatible=close,
                                            max_abs=float((tensor.double() - old.double()).abs().max()))
                    if not close:
                        raise RuntimeError(f'fixture static geometry mismatch: {key}')
                loaded[key] = tensor
            # Wall/ghost rows stay attached to the same stationary tank.
            fixed = state.kinds != 0
            boundary_error = float((loaded['positions'][fixed] - state.positions[fixed]).abs().max())
            setup_checks['fixed_positions_max_abs'] = boundary_error
            if boundary_error > 2e-7:
                raise RuntimeError(f'fixture fixed geometry moved: {boundary_error}')
            system.state = dataclasses.replace(state, **loaded, pressures=None, soundspeeds=None,
                                               surfaceIndicators=None, surfaceNormals=None, surfaceLambdas=None)
            system.adjacency = None
            system.t = torch.tensor(fixture_time, device=ctx.device, dtype=ctx.dtype)
            ctx.config.dt = torch.tensor(1e-4, device=ctx.device, dtype=ctx.dtype)
            if original_post:
                original_post(ctx, system, 0)  # set roll/gravity at fixture time before the first step
            physical_step = 0

        def post(ctx, state, i):
            nonlocal physical_step
            if original_post:
                original_post(ctx, state, i)
            physical_step = i + 1

        case = dataclasses.replace(case, buildSystem=build, initialConditions=initial, postStep=post)
        kw.update(params={'rollDataFile': str((args.fixtures / 'roll_history.txt').resolve())})
        if args.video:
            kw.update(plot=True, video=True, show=False, asyncPlot=False, plotBackend='vispy',
                      plotBackendOptions={'app_backend': 'egl'}, plotInterval=50,
                      exportRoot='/tmp/warpsph-sloshing-video')
        run_result = original_run(case, **kw)
        return run_result

    runner.run = run
    scenario = 'sloshing-' + args.phase
    benchmark.SCENARIOS[scenario] = ('sloshingTank', 'deltaSPH', 200, 'graph')
    args.scenario = scenario
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.worker(args)
    torch.cuda.synchronize()
    for item in observations:
        if 'events' in item:
            begin, end = item.pop('events')
            item['device_interval_ms'] = begin.elapsed_time(end)
    report = json.loads(args.output.read_text())
    report.update(fixture_phase=args.phase, fixture_time=fixture_time,
                  fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                  fixture_manifest_sha256=hashlib.sha256((args.fixtures / 'manifest.json').read_bytes()).hexdigest(),
                  fixture_static_checks=setup_checks, section_observations=observations,
                  instrumented_diagnostic=args.measure_search, video_validation=args.video,
                  video_path=run_result.videoPath, image_path=run_result.ctx.imagePath,
                  fixture_scope=__doc__)
    report['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    benchmark.save(args.output, report)


def suite(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports, failures = [], []
    for phase in args.phases:
        for repeat in range(args.repeats):
            policies = args.policies[repeat % len(args.policies):] + args.policies[:repeat % len(args.policies)]
            for policy in policies:
                tag = '-diagnostic' if args.measure_search else '-video' if args.video else ''
                path = args.output_dir / f'sloshing-{phase}-{policy}{tag}-r{repeat}.json'
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--phase', phase,
                           '--policy', policy, '--output', str(path), '--fixtures', str(args.fixtures),
                           '--warmup', str(args.warmup), '--active', str(args.active)]
                if args.pipeline:
                    command.append('--pipeline')
                if args.measure_search:
                    command.append('--measure-search')
                if args.video:
                    command.append('--video')
                print(f'START {phase} {policy} r{repeat}{tag}', flush=True)
                with path.with_suffix('.log').open('w') as log:
                    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if done.returncode:
                    failures.append(str(path.with_suffix('.log')))
                    print(f'FAILED {failures[-1]}', flush=True)
                else:
                    r = json.loads(path.read_text())
                    reports.append(r)
                    count = sum(c['section'] == 'search' and c['step'] >= args.warmup for c in r['section_observations'])
                    print(f'DONE {r["mean_loop_ms"]:.3f} ms/step, {count}/{args.active} CSR rebuilds, graph={r["graph"]}', flush=True)
                benchmark.save(args.output_dir / 'progress.json', dict(completed=len(reports), failures=failures))
    comparisons = benchmark.compare(reports)
    benchmark.save(args.output_dir / 'summary.json', dict(complete=not failures, failures=failures,
                                                        comparisons=comparisons, command=sys.argv))
    if failures:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--phase', choices=['startup', 'impact1', 'impact3'])
    ap.add_argument('--policy', choices=benchmark.POLICIES)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--phases', nargs='+', default=['startup', 'impact1', 'impact3'])
    ap.add_argument('--policies', nargs='+', default=['wide-fused', 'narrow-fused', 'narrow-split', 'automatic'])
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'docs/regression/sloshing_motion_2026_10_15/trials')
    ap.add_argument('--fixtures', type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--active', type=int, default=1000)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--pipeline', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--measure-search', action='store_true')
    ap.add_argument('--video', action='store_true')
    args = ap.parse_args()
    worker(args) if args.worker else suite(args)


if __name__ == '__main__':
    main()
