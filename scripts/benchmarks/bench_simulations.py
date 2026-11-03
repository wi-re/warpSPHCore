"""Run the released frontend against Warp or native core operators.

Set PYTHONPATH to the paired core/frontend/integrator source roots. Each backend
must run in a fresh process; --operators selects only core pair operators and
CRK accumulation passes. Frontend-specific kernels and Verlet search stay Warp.
The runner's step timer includes integrator stages, search and layout work;
setup, JIT compilation, output and diagnostics are outside the measured steps.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def revision(module):
    source = Path(module.__file__).resolve()
    result = subprocess.run(['git', '-C', str(source.parent), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    digest = hashlib.sha256()
    for path in sorted(source.parent.rglob('*')):
        if path.suffix in ('.py', '.h', '.cpp', '.cu', '.cuh'):
            digest.update(str(path.relative_to(source.parent)).encode())
            digest.update(path.read_bytes())
    return dict(source=str(source), commit=result.stdout.strip(), source_sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operators', choices=('warp', 'native'), required=True)
    parser.add_argument('--case', choices=('waveEquation', 'tgv-wc', 'tgv', 'linearWave', 'gresho'), required=True)
    parser.add_argument('--scheme')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--dim', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--n-h', type=float, default=4.)
    parser.add_argument('--order', choices=('input', 'morton'), default='morton')
    parser.add_argument('--dt', type=float, default=1.e-4)
    parser.add_argument('--sound-speed', type=float, help='Pin Delta-SPH sound speed independently of dt; otherwise the upstream case selects it')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--profile', action='store_true', help='Record dispatched Torch operators; these timings are not benchmark results')
    parser.add_argument('--stage-profile', action='store_true', help='Partition runner time by frontend stage and actual Warp/native dispatch (adds event overhead)')
    parser.add_argument('--snapshots', action='store_true', help='Save every accepted state for trajectory comparison, outside step timing')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error('steps and warmup must be positive')
    if args.case == 'waveEquation' and args.order != 'input':
        parser.error('the upstream wave runner has no Morton lifecycle adapter; use --order input')
    os.environ['WARPSPHCORE_BACKEND'] = 'warp'
    os.environ['WARPSPHCORE_OPERATORS'] = args.operators
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import torch
    import warpSPH
    import warpSPHCore
    import warpSPHIntegrators
    from warpSPH.runner import run
    from warpSPH.cases import importAll
    from warpSPH.runner.case import getCase

    torch.set_num_threads(args.threads)
    torch.manual_seed(173)
    importAll()
    case = getCase(args.case)
    args.out.mkdir(parents=True, exist_ok=True)
    snapshots, gaps, adjacency_objects = [], [], []
    profiler = None
    measured_start = measured_end = None
    last_boundary = None
    initial_hook, previous_hook = case.initialConditions, case.postStep

    def state_values(system):
        state = system.state
        # UID order removes the storage permutation without hiding dynamics.
        order = torch.argsort(state.UIDs)
        names = ('UIDs', 'positions', 'supports', 'masses', 'kinds', 'velocities',
                 'densities', 'pressures', 'internalEnergies', 'u', 'v')
        values = {name: value.detach()[order].cpu().clone() for name in names
                  if isinstance(value := getattr(state, name, None), torch.Tensor)}
        for name, value in values.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise AssertionError(f'Non-finite accepted {name}')
        return values

    stage_timing = None
    if args.stage_profile:
        if args.profile:
            parser.error('Use separate runs for --profile and --stage-profile')
        from stage_timing import StageTiming
        stage_timing = StageTiming(args.device).install()

    initial_values = None

    def initial(ctx, system):
        nonlocal initial_values
        if initial_hook:
            initial_hook(ctx, system)
        initial_values = state_values(system)

    def post(ctx, system, step):
        nonlocal last_boundary, measured_start, measured_end
        if previous_hook:
            previous_hook(ctx, system, step)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        now = time.perf_counter()
        if step == args.warmup - 1:
            measured_start = time.time()
        if step == args.warmup + args.steps - 1:
            measured_end = time.time()
        # Consecutive accepted-boundary intervals also include runner outputs
        # and diagnostics, unlike the runner's integrator/layout timer.
        if step >= args.warmup and last_boundary is not None:
            gaps.append(1000 * (now - last_boundary))
        last_boundary = now
        adjacency_objects.append(id(system.adjacency))
        if args.snapshots:
            snapshots.append(state_values(system))
        if profiler is not None:
            if step == args.warmup - 1:
                profiler.start()
            elif step == args.warmup + 1:
                profiler.stop()

    case = replace(case, initialConditions=initial, postStep=post)
    params = {}
    if args.case == 'tgv-wc':
        params = dict(shuffleIters=0, jitter=0., shifting=False, targetDt=args.dt)
        if args.sound_speed is not None:
            params['soundSpeed'] = args.sound_speed
    elif args.case == 'tgv':
        params = dict(relaxSteps=0, jitter=0., shifting=False)
    options = dict(device=args.device, nx=args.nx, dim=args.dim, n_h=args.n_h,
                   dt=args.dt, adaptiveDt=False, nSteps=args.warmup + args.steps,
                   kernel='Wendland4', quiet=True, progress=False, plot=False,
                   video=False, store=False, cudaGraph=False, params=params,
                   particleOrder=args.order)
    if args.scheme:
        options['scheme'] = args.scheme
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if args.steps < 2:
            parser.error('--profile requires at least two measured steps')
        profiler = torch.profiler.profile(activities=activities)
        result = run(case, **options)
        averages = profiler.key_averages()
        events = {e.key: e.count for e in averages
                  if 'warpsph_native::' in e.key}
        (args.out / 'native-dispatch.json').write_text(json.dumps(events, indent=2))
        (args.out / 'profile-cpu.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=30))
        if args.device == 'cuda':
            (args.out / 'profile-gpu.txt').write_text(averages.table(sort_by='self_device_time_total', row_limit=30))
    else:
        result = run(case, **options)
    if stage_timing is not None:
        stage_timing.save(args.out / 'stage-timing.json', args.warmup, args.steps)
        stage_timing.close()
    if result.diverged:
        raise AssertionError(f'Simulation diverged: {result.stopReason}')
    rows = [row for row in result.trajectory if row['step'] >= args.warmup]
    timings = [row['stepTime_ms'] for row in rows]
    if len(timings) != args.steps:
        raise AssertionError(f'Expected {args.steps} measured steps, got {len(timings)}')
    torch.save(dict(initial=initial_values, final=state_values(result.state),
                    snapshots=snapshots), args.out / 'states.pt')
    metadata = dict(arguments={**vars(args), 'out': str(args.out)},
                    versions=dict(core=warpSPHCore.__version__, integrators=warpSPHIntegrators.__version__,
                                  torch=torch.__version__),
                    sources={name: revision(module) for name, module in
                             [('core', warpSPHCore), ('frontend', warpSPH), ('integrators', warpSPHIntegrators)]},
                    particles=len(result.state.state.positions),
                    csr_candidates_per_particle=float(result.state.adjacency.numNeighbors.float().mean().cpu()),
                    median_step_ms=statistics.median(timings),
                    mean_step_ms=statistics.mean(timings),
                    median_runner_interval_ms=statistics.median(gaps),
                    mean_runner_interval_ms=statistics.mean(gaps),
                    step_times_ms=timings, runner_intervals_ms=gaps,
                    adjacency_changes=sum(a != b for a, b in zip(adjacency_objects, adjacency_objects[1:])),
                    final_time=float(result.state.t), profiled=args.profile,
                    measured_window_unix=(measured_start, measured_end),
                    trajectory=result.trajectory)
    if args.device == 'cuda':
        metadata['gpu'] = torch.cuda.get_device_name()
        metadata['peak_torch_bytes'] = torch.cuda.max_memory_allocated()
    metadata['operator_backend'] = warpSPHCore.operator_backend
    (args.out / 'result.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: metadata[key] for key in ('particles', 'csr_candidates_per_particle', 'median_step_ms', 'median_runner_interval_ms')}))


if __name__ == '__main__':
    main()
