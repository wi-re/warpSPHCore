#!/usr/bin/env python3
"""CG timing on a periodic Warp stencil or fixed-neighborhood SPH action.

Use --source-root to compare an archived src tree. Compilation/warmup is
excluded. Each sample includes final synchronization; device intervals are
reported separately. This is a loop benchmark, not an SPH trajectory claim.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

import torch
import warp as wp


@wp.kernel
def stencil(x: wp.array(dtype=Any), side: int, y: wp.array(dtype=Any)):
    i = wp.tid()
    row, col = i // side, i % side
    left = row * side + (col + side - 1) % side
    right = row * side + (col + 1) % side
    up = ((row + side - 1) % side) * side + col
    down = ((row + 1) % side) * side + col
    y[i] = x.dtype(4.01) * x[i] - x[left] - x[right] - x[up] - x[down]


def timeMatvec(matvec, b, *, repeat, batch):
    """Public-call latency and warmed graph throughput, without a profiler.

    Graph timing includes all kernels in the action, including the SPH Torch
    scale/add. Repeated calls read the same vector and write disposable output;
    this measures a hot, fixed adjacency, not a changing simulation trajectory.
    CUDA event intervals include device idle gaps and are not busy-kernel sums.
    """
    from warpSPHCore.autograd.stream import torchWarpCapture, torchWarpStream

    output = matvec(b)
    torch.cuda.current_stream().synchronize()
    walls, intervals = [], []
    for _ in range(repeat):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        output = matvec(b)
        stop.record()
        stop.synchronize()
        walls.append((time.perf_counter() - before) * 1e3)
        intervals.append(start.elapsed_time(stop))
    expected = output
    graph = torch.cuda.CUDAGraph()
    with torchWarpStream(b.device), torchWarpCapture(graph):
        for _ in range(batch):
            output = matvec(b)
    graph.replay()
    torch.cuda.current_stream().synchronize()
    graphWalls, graphIntervals = [], []
    for _ in range(repeat):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        graph.replay()
        stop.record()
        stop.synchronize()
        graphWalls.append((time.perf_counter() - before) * 1e3 / batch)
        graphIntervals.append(start.elapsed_time(stop) / batch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    return dict(batch=batch, eager_wall_ms=walls, eager_device_interval_ms=intervals,
                median_eager_wall_ms=statistics.median(walls),
                graph_wall_ms_per_call=graphWalls, graph_device_ms_per_call=graphIntervals,
                median_graph_device_ms_per_call=statistics.median(graphIntervals),
                median_graph_wall_ms_per_call=statistics.median(graphWalls),
                finite_output=bool(torch.isfinite(output).all()), matches_eager_output=True,
                scope='fixed input and adjacency; public matvec including scale/add; warmed batch graph')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', type=int, nargs='+', default=[1024, 16384, 65536])
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--check-every', type=int, default=8)
    parser.add_argument('--engine', choices=['torch', 'native', 'native-cached', 'warp-eager', 'warp-internal', 'warp-cached'], default='torch')
    parser.add_argument('--workload', choices=['stencil', 'sph'], default='stencil')
    parser.add_argument('--neighbors', type=int, default=55, help='requested SPH neighbors; actual CSR degree is recorded')
    parser.add_argument('--matvec-batch', type=int, default=0, help='also time isolated matvec calls and a graph batch of this size')
    parser.add_argument('--rtol', type=float, default=0)
    parser.add_argument('--warm-start', action='store_true')
    parser.add_argument('--no-history', action='store_true')
    parser.add_argument('--profile', action='store_true', help='separate instrumented pass after steady timing')
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2] / 'src')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('an accessible CUDA device is required')
    if min(args.iterations, args.repeat, args.warmup, args.check_every, *args.ns) <= 0:
        parser.error('sizes, budgets and sample counts must be positive')
    if args.no_history and not args.engine.startswith('native'):
        parser.error('--no-history requires a native engine')
    if not math.isfinite(args.rtol) or args.rtol < 0:
        parser.error('--rtol must be finite and nonnegative')
    if args.neighbors <= 0 or args.matvec_batch < 0:
        parser.error('--neighbors must be positive and --matvec-batch nonnegative')
    sys.path.insert(0, str(args.source_root.resolve()))
    import warpSPHCore
    from warpSPHCore.linalg import cg
    if args.engine.startswith('native'):
        from warpSPHCore.linalg import CGWorkspace

    torch.set_num_threads(2)
    wp.init()
    records = []
    for n in args.ns:
        torch.cuda.reset_peak_memory_stats()
        side = math.isqrt(n)
        if side * side != n:
            parser.error('each N must be a square')
        g = torch.Generator(device='cuda').manual_seed(0)
        dtype = warpSPHCore.type_config.get_torch_precision()
        b = torch.randn(n, device='cuda', generator=g, dtype=dtype)
        inverse = torch.full_like(b, 1 / 4.01)
        calls = 0

        def matvecOut(v, out):
            nonlocal calls
            calls += 1
            wp.launch(stencil, dim=n, inputs=[wp.from_torch(v), side, wp.from_torch(out)],
                      stream=wp.get_stream() if wp.get_stream().cuda_stream == torch.cuda.current_stream(v.device).cuda_stream
                      else wp.stream_from_torch(torch.cuda.current_stream(v.device)))

        def matvec(v):
            out = torch.empty_like(v)
            matvecOut(v, out)
            return out

        neighborhood = None
        if args.workload == 'sph':
            from warpSPHCore import OperationProperties, ParticleState, radiusSearchCompactHashMap, warpOperation
            from warpSPHCore.enumTypes import GradientScheme, KernelFunctions, LaplacianScheme, SupportScheme, WarpOperation
            from warpSPHCore.util import generateNeighborTestData
            torch.cuda.current_stream().synchronize()
            neighborStart = time.perf_counter()
            positions, supports, _, domain, dx = generateNeighborTestData(side, args.neighbors, 2, True, b.device)
            if positions.shape[0] != n:
                raise RuntimeError('generated lattice size differs from requested N')
            particles = ParticleState(positions=positions, supports=supports,
                                      masses=torch.full_like(b, dx ** 2), densities=torch.ones_like(b),
                                      kinds=torch.zeros(n, dtype=torch.int32, device=b.device))
            adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
            torch.cuda.current_stream().synchronize()
            neighborSetupMs = (time.perf_counter() - neighborStart) * 1e3
            degree = adjacency.numNeighbors
            edges = int(degree.sum())
            neighborhood = dict(requested=args.neighbors, edges=edges,
                                mean=edges / n, minimum=int(degree.min()), maximum=int(degree.max()),
                                self_edges=n, nonself_edges=edges - n,
                                padded_pair_capacity=adjacency.paddedJ.numel(),
                                pair_index_bytes=(adjacency.paddedI.numel() * adjacency.paddedI.element_size()
                                                  + adjacency.paddedJ.numel() * adjacency.paddedJ.element_size()),
                                support_over_spacing=float(supports[0]) / dx,
                                setup_ms=neighborSetupMs,
                                scope='CSR entries including self; no Verlet skin; lattice distance checks still occur')
            props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Laplacian,
                                        gradientMode=GradientScheme.Difference, laplacianMode=LaplacianScheme.Brookshaw,
                                        supportMode=SupportScheme.Gather)

            def matvec(v):
                nonlocal calls
                calls += 1
                lap = warpOperation(particles, props, domain, adjacency=adjacency, queryValues=v, referenceValues=v)
                return v - dx ** 2 * lap

            inverse = torch.ones_like(b)
        setupStart = time.perf_counter()
        workspace = CGWorkspace(b, args.iterations, history=not args.no_history,
                                capture=args.engine == 'native-cached') if args.engine.startswith('native') else None
        workspaceSetupMs = (time.perf_counter() - setupStart) * 1e3 if workspace is not None else None
        x0 = None
        if args.warm_start:
            x0, b = b * .9, matvec(b)
        # Numerical checks outside timing: the periodic equal-volume SPH
        # action should be symmetric positive definite on this lattice.
        probe = torch.randn(n, device='cuda', generator=g, dtype=dtype)
        ab, ap = matvec(b), matvec(probe)
        symmetry = float((torch.dot(b, ap) - torch.dot(ab, probe)).abs() /
                         (b.norm() * ap.norm() + ab.norm() * probe.norm()).clamp_min(1e-30))
        if not torch.dot(b, ab) > 0 or symmetry > 100 * torch.finfo(dtype).eps:
            raise RuntimeError('benchmark action did not pass its SPD probe')

        upstream = None
        if args.engine.startswith('warp-'):
            from _warp_cg_reference import WarpCGReference
            upstream = WarpCGReference(matvec, b, maxiter=args.iterations, rtol=args.rtol,
                                       precond=inverse, schedule=args.engine.removeprefix('warp-'),
                                       checkEvery=args.check_every,
                                       matvecOut=matvecOut if args.workload == 'stencil' else None)

        def solve():
            if upstream is not None:
                if upstream.schedule == 'cached' and upstream.graph is None:
                    upstream.prepare(b, x0)
                return upstream.observe(b, x0)
            kw = dict(workspace=workspace, matvecOut=matvecOut if args.workload == 'stencil' else None) if workspace is not None else {}
            return cg(matvec, b, x0, precond=inverse, maxiter=args.iterations, rtol=args.rtol,
                      atol=0, checkEvery=args.check_every, **kw)

        before = time.perf_counter()
        solve()
        torch.cuda.current_stream().synchronize()
        firstSolveMs = (time.perf_counter() - before) * 1e3
        for _ in range(args.warmup - 1):
            solve()
        torch.cuda.synchronize()
        walls, devices, matvecs = [], [], []
        for _ in range(args.repeat):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            calls = 0
            start.record()
            before = time.perf_counter()
            result = solve()
            stop.record()
            stop.synchronize()
            walls.append((time.perf_counter() - before) * 1e3)
            devices.append(start.elapsed_time(stop))
            matvecs.append(getattr(result, 'matvecs', calls) if upstream is None else result.scheduled + 2)
        true = float((b - matvec(result.x)).norm() / b.norm())
        status = result.statusInt() if upstream is None else ('verified' if result.verified else 'unverified')
        if args.rtol and ((result.statusInt() < 0 if upstream is None else not result.verified) or true > args.rtol):
            raise RuntimeError(f'solve-to-tolerance failed: status={status}, true relative residual={true}')
        profile = None
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA], profile_memory=True) as prof:
                solve()
                torch.cuda.current_stream().synchronize()
            events = prof.events()
            with tempfile.TemporaryDirectory(prefix='cg-profile-') as directory:
                trace = Path(directory) / 'trace.json'
                prof.export_chrome_trace(str(trace))
                activity = json.loads(trace.read_text())['traceEvents']
            kernels = [e for e in activity if e.get('cat') == 'kernel']
            profile = dict(cuda_events=sum(e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') for e in activity),
                           kernel_events=len(kernels),
                           summed_kernel_ms=sum(e.get('dur', 0) for e in kernels) / 1e3,
                           count_scope='emitted profiler events; conditional loop repetitions may be collapsed',
                           torch_local_scalar_reads=sum(e.name == 'aten::_local_scalar_dense' for e in events),
                           torch_empty_calls=sum(e.name in ('aten::empty', 'aten::empty_strided') for e in events))
        solveMemory = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                          peak_allocated=torch.cuda.max_memory_allocated())
        isolated = timeMatvec(matvec, b, repeat=args.repeat, batch=args.matvec_batch) if args.matvec_batch else None
        records.append(dict(n=n, iterations=args.iterations, check_every=args.check_every,
                            neighborhood=neighborhood, isolated_matvec=isolated,
                            solve_memory_bytes=solveMemory,
                            median_wall_ms=statistics.median(walls), wall_ms=walls,
                            median_device_interval_ms=statistics.median(devices), device_interval_ms=devices,
                            matvecs=matvecs, status=status, true_relative_residual=true,
                            matvec_count_source='solver_counts' if upstream is None else 'scheduled_plus_initial_and_verification',
                            completed_updates=int(result.iterations) if getattr(result, 'iterations', None) is not None else None,
                            scheduled_updates=getattr(result, 'loopIterations', None) if upstream is None else result.scheduled,
                            verifications=getattr(result, 'verifications', None), restarts=getattr(result, 'restarts', None), symmetry_probe=symmetry,
                            workspace_bytes=workspace.nbytes if workspace is not None else None,
                            workspace_host_setup_ms=workspaceSetupMs,
                            first_solve_ms=firstSolveMs,
                            capture_builds=workspace.captureBuilds if workspace is not None else None,
                            capture_ms=workspace.captureMilliseconds if workspace is not None else None,
                            setup_iterations=getattr(result, 'setupIterations', None),
                            upstream_setup_ms=upstream.setup_ms if upstream is not None else None,
                            upstream_capture_ms=upstream.capture_ms if upstream is not None else None,
                            instrumented_pass=profile,
                            upstream_recurrence_residual=result.recurrence if upstream is not None else None))
        from _work_metrics import cgWork
        records[-1]['work'] = cgWork(records[-1], engine=args.engine)
        records[-1]['preconditioner_applications'] = getattr(result, 'preconditionerApplications', None)
    report = dict(implementation=str(Path(warpSPHCore.__file__).resolve()), torch=torch.__version__,
                  warp=wp.__version__, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                  dtype=str(dtype), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  warmup=args.warmup, repeat=args.repeat, records=records,
                  engine=args.engine, workload=args.workload, rtol=args.rtol, warm_start=args.warm_start,
                  history=not args.no_history if upstream is None else False)
    report['compile_glue'] = os.environ.get('WARPSPHCORE_COMPILE_GLUE', '0')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    # CUDA's legacy default stream cannot be captured by Warp's internal
    # scheduler. Use the same nondefault stream for every compared backend.
    if torch.cuda.is_available():
        runStream = torch.cuda.Stream()
        with torch.cuda.stream(runStream):
            main()
        runStream.synchronize()
    else:
        main()
