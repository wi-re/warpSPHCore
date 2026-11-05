"""Run the released frontend against Warp or native core operators.

Set PYTHONPATH to the paired core/frontend/integrator source roots. Each backend
must run in a fresh process; --operators selects only core pair operators and
CRK accumulation passes. --frontend-operators independently selects converted
frontend families; actual dispatch still identifies mixed execution.
The runner's step timer includes integrator stages, search and layout work;
setup, JIT compilation, output and diagnostics are outside the measured steps.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import re


def fixed_step_clock_status(trajectory, dt, precision='float32'):
    """Detect material step clamps; retain small precision/safety margins."""
    rows = [row for row in trajectory if 't' in row]
    deviations = [(row['step'], float(row['t']) - float(previous['t']) - dt)
                  for previous, row in zip(rows, rows[1:])]
    epsilon = 2. ** (-23 if precision == 'float32' else -52)
    relative_tolerance = 32 * epsilon
    mismatches = [step for step, error in deviations
                  if not math.isfinite(error) or abs(error) > max(1.e-15, abs(dt) * relative_tolerance)]
    return dict(verified=bool(deviations) and not mismatches,
                requested_dt=dt, relative_dt_tolerance=relative_tolerance,
                mismatched_steps=mismatches,
                maximum_dt_deviation=max((abs(error) for _, error in deviations), default=None))


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


def frontend_gpu_policies(module):
    """Source-declared widths; instrumented GPU events establish selected kernels."""
    package = Path(module.__file__).parent
    policies = {}
    for manifest in sorted(package.rglob('native/sources.json')):
        for filename in json.loads(manifest.read_text()).get('gpu', ()):
            source = manifest.parent / filename
            text = source.read_text()
            widths = sorted(set(map(int, re.findall(r'CooperativeGpuLauncher<(\d+)>', text))))
            policies[str(source.relative_to(package))] = dict(cooperative_lanes=widths,
                pointwise_launcher='GpuLauncher{' in text, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operators', choices=('warp', 'native'), required=True)
    parser.add_argument('--frontend-operators', choices=('warp', 'native'), default='warp',
                        help='Converted frontend families only; remaining stages may still use Warp')
    parser.add_argument('--search-backend', choices=('auto', 'warp', 'native'), default='auto', help='Neighborhood implementation; auto follows core operators')
    parser.add_argument('--case', choices=('waveEquation', 'tgv-wc', 'tgv', 'linearWave', 'gresho', 'sedov', 'drivenTurbulence'), required=True)
    parser.add_argument('--scheme')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--dim', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--n-h', type=float, default=4.)
    parser.add_argument('--kernel', default='Wendland4', help='Explicit physical kernel policy; defaults to Wendland4')
    parser.add_argument('--order', choices=('input', 'morton'), default='morton')
    parser.add_argument('--dt', type=float, default=1.e-4)
    parser.add_argument('--sound-speed', type=float, help='Pin Delta-SPH sound speed independently of dt; otherwise the upstream case selects it')
    parser.add_argument('--ambient-energy', type=float, help='Sedov initial uniform specific internal energy; otherwise preserve the case default')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--profile', action='store_true', help='Record dispatched Torch operators; these timings are not benchmark results')
    parser.add_argument('--stage-profile', action='store_true', help='Partition runner time by frontend stage and actual Warp/native dispatch (adds event overhead)')
    parser.add_argument('--snapshots', action='store_true', help='Save every accepted state for trajectory comparison, outside step timing')
    parser.add_argument('--storage-chunks', type=int, choices=(0, 1, 2, 4), default=0,
                        help='0: CSR; otherwise bound self rows into the requested few chunks')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error('steps and warmup must be positive')
    if args.ambient_energy is not None and (args.case != 'sedov' or args.ambient_energy < 0):
        parser.error('--ambient-energy requires the Sedov case and a nonnegative value')
    if args.case == 'waveEquation' and args.order != 'input':
        parser.error('the upstream wave runner has no Morton lifecycle adapter; use --order input')
    os.environ['WARPSPHCORE_BACKEND'] = 'warp'
    os.environ['WARPSPHCORE_OPERATORS'] = args.operators
    os.environ['WARPSPH_FRONTEND_OPERATORS'] = args.frontend_operators
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
    if args.storage_chunks:
        import math
        from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList
        from warpSPHCore import SupportScheme, get_torch_index_dtype
        original_build = case.buildSystem

        def build_chunked(ctx):
            def factory(state, config):
                from warpSPH.configurations.compute import neighborSearchBackend
                n, dimension = state.positions.shape
                width = dimension if args.scheme in ('CRKSPH', 'CompSPH') else 1
                index_bytes = torch.empty(0, dtype=get_torch_index_dtype()).element_size()
                policy = RowChunkPolicy(edgeBudget=2**31-1,
                    maxRows=math.ceil(n/args.storage_chunks),
                    bytesPerEdge=index_bytes + width*state.positions.element_size(),
                    maxElementsPerEdge=1 if args.frontend_operators=='native' and args.scheme in ('CRKSPH', 'CompSPH') else width)
                return buildChunkedVerletList(state, config.domain, policy=policy,
                    supportMode=SupportScheme.SuperSymmetric,
                    verletScale=config.verletScale, storeI=False, backend=neighborSearchBackend(config))
            import inspect
            if 'neighborhoodFactory' in inspect.signature(original_build).parameters:
                return original_build(ctx, neighborhoodFactory=factory)
            # Some released case samplers lack the construction hook. This
            # setup-only conversion is excluded from timing and recorded below.
            system = original_build(ctx)
            if args.scheme in ('MFM', 'MFV'):
                # Meshless geometry is rebuilt at every stage. Keep the bounded
                # factory on clone lineage, rather than timing a CSR fallback
                # after an initial chunked system.adjacency is discarded.
                system.state._meshlessNeighborhoodFactory = factory
                system.state._meshlessGeometry = None
            system.adjacency = factory(system.state, ctx.config)
            return system
        case = replace(case, buildSystem=build_chunked)
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
                 'densities', 'pressures', 'internalEnergies', 'u', 'v', 'Q',
                 'totalEnergies', 'materials', 'entropies', 'soundspeeds', 'alphas')
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

    def active_adjacency(system):
        adjacency = getattr(system, 'adjacency', None)
        if adjacency is not None:
            return adjacency, 'system.adjacency'
        context = getattr(system, 'meshlessContext', None)
        geometry = getattr(context, 'geom', None)
        if geometry is not None:
            return geometry.adjacency, 'meshlessContext.geom'
        ticket = getattr(system.state, '_meshlessGeometry', None)
        if ticket is not None:
            return ticket.geometry.adjacency, 'accepted geometry ticket'
        raise ValueError('Benchmark needs an explicit active neighborhood for this solver')

    initial_values = None
    initial_uid_order = None

    def initial(ctx, system):
        nonlocal initial_values, initial_uid_order
        if initial_hook:
            initial_hook(ctx, system)
        initial_values = state_values(system)
        initial_uid_order = system.state.UIDs.detach().cpu().clone()

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
        active = active_adjacency(system)[0]
        if args.storage_chunks and len(getattr(active, 'chunks', ())) != args.storage_chunks:
            raise AssertionError('Requested row-chunk storage did not survive the accepted-stage lifecycle')
        adjacency_objects.append(id(active))
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
    elif args.case == 'sedov' and args.ambient_energy is not None:
        params['uAmbient'] = args.ambient_energy
    options = dict(device=args.device, nx=args.nx, dim=args.dim, n_h=args.n_h,
                   dt=args.dt, adaptiveDt=False, nSteps=args.warmup + args.steps,
                   kernel=args.kernel, quiet=True, progress=False, plot=False,
                   video=False, store=False, cudaGraph=False, params=params,
                   particleOrder=args.order, neighborBackend=args.search_backend)
    if args.case == 'drivenTurbulence':
        # Preserve this case's adaptive source/timestep contract. A sufficiently
        # small common cap gives matched accepted steps, verified in snapshots.
        options['adaptiveDt'] = True
        options['maxDt'] = args.dt
        options['integrationScheme'] = 'rungeKutta2'
        options['storeMode'] = 'trajectory'  # No checkpoint export during timing-only runs.
        params.update(gamma=1.4, accelerationRMS=.1, thermalModel='adiabatic',
                      soundSpeed=args.sound_speed if args.sound_speed is not None else 1.)
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
                  if 'warpsph_native::' in e.key or 'warpsph_frontend::' in e.key}
        (args.out / 'native-dispatch.json').write_text(json.dumps(events, indent=2))
        kernels = [dict(name=e.key, count=e.count) for e in averages
                   if str(e.device_type).upper().endswith('CUDA')]
        (args.out / 'observed-gpu-kernels.json').write_text(json.dumps(kernels, indent=2))
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
    adjacency, adjacency_source = active_adjacency(result.state)
    metadata = dict(arguments={**vars(args), 'out': str(args.out)},
                    versions=dict(core=warpSPHCore.__version__, integrators=warpSPHIntegrators.__version__,
                                  torch=torch.__version__, warp=__import__('warp').__version__),
                    sources={name: revision(module) for name, module in
                             [('core', warpSPHCore), ('frontend', warpSPH), ('integrators', warpSPHIntegrators)]},
                    particles=len(result.state.state.positions),
                    csr_candidates_per_particle=float(adjacency.numNeighbors.float().mean().cpu()),
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
    metadata['frontend_operator_backend'] = args.frontend_operators
    metadata['case_parameters'] = params
    metadata['time_policy'] = ('adaptive source contract with common maxDt cap' if args.case == 'drivenTurbulence'
                               else 'fixed accepted timestep')
    if args.case != 'drivenTurbulence':
        metadata['fixed_step_clock'] = fixed_step_clock_status(result.trajectory, args.dt, args.precision)
        metadata['performance_window_qualified'] = metadata['fixed_step_clock']['verified']
        if not metadata['performance_window_qualified']:
            metadata['time_policy'] = 'requested fixed timestep was not preserved; timings are unqualified'
    final_uid_order = result.state.state.UIDs.detach().cpu()
    metadata['particle_order'] = dict(
        requested=args.order, saved_state_order='ascending UID',
        initial_storage_sha256=hashlib.sha256(initial_uid_order.numpy().tobytes()).hexdigest(),
        final_storage_sha256=hashlib.sha256(final_uid_order.numpy().tobytes()).hexdigest(),
        storage_order_preserved=torch.equal(initial_uid_order, final_uid_order))
    metadata['execution_scope'] = 'uninstrumented; use matched stage profile and source coverage ledger'
    if args.stage_profile:
        from collections import Counter
        observed = Counter()
        for frame in json.loads((args.out / 'stage-timing.json').read_text())['frames']:
            observed.update(frame['calls'])
        metadata['observed_numerical_dispatch'] = dict(observed)
        has_warp = any(key.startswith('Warp:') and count for key, count in observed.items())
        has_native = any(key.startswith('native:') and count for key, count in observed.items())
        metadata['execution_scope'] = ('observed mixed Warp/native kernels' if has_warp and has_native else
            'observed Warp numerical kernels' if has_warp else
            'observed native kernels with Torch/Python controller; no Warp launches' if has_native else
            'Torch/Python controller; no numerical kernel launches observed')
    metadata['cpu_execution'] = dict(torch_threads=torch.get_num_threads(),
                                    affinity=sorted(os.sched_getaffinity(0)))
    metadata['frontend_gpu_policies'] = frontend_gpu_policies(warpSPH)
    metadata['core_gpu_policies'] = frontend_gpu_policies(warpSPHCore)
    metadata['kernel_policy_scope'] = ('Source-declared native launch widths with SHA256; the separate --profile run records actual GPU kernel names. Warp per-operator lane defaults are pinned in the source; explicit process override is recorded below.')
    metadata['native_build_environment'] = {name: os.environ.get(name) for name in
        ('WARPSPHCORE_NATIVE_BUILD', 'WARPSPHCORE_NATIVE_CPU_ARCH', 'TORCH_CUDA_ARCH_LIST',
         'WARPSPHCORE_NEIGHBOR_LANES', 'MAX_JOBS')}
    from dataclasses import asdict
    metadata['neighborhood'] = dict(kind=type(adjacency).__name__, source=adjacency_source, search_backend=adjacency.searchBackend,
        num_edges=adjacency.numEdges,
        chunks=[asdict(chunk.metadata) for chunk in getattr(adjacency, 'chunks', ())],
        policy=asdict(adjacency.plan.policy) if hasattr(adjacency, 'plan') else None,
        setup_conversion_may_use_csr=bool(args.storage_chunks and
            'neighborhoodFactory' not in __import__('inspect').signature(original_build).parameters))
    (args.out / 'result.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: metadata[key] for key in ('particles', 'csr_candidates_per_particle', 'median_step_ms', 'median_runner_interval_ms')}))


if __name__ == '__main__':
    main()
