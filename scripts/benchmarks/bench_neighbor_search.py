#!/usr/bin/env python3
"""Neighbor-search scaling: public rebuilds, fixed-grid builds and bare grid passes.

Steady wall/event samples have no profiler/instrumentation. Separate diagnostic
scope intervals include their event overhead. Graph batches time preallocated
count/collect kernels, not a capturable variable-size CSR search. Uniform and
jittered/shuffled periodic particles have equal constant support radii.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]


def candidateVisits(positions, grid, periodic):
    """Exact cell-occupancy visits before distance filtering; no timing instrumentation.

    Uses the built grid's cell IDs and occupancy, including self. This counts
    particles, not hash-bucket probes; repeated wrapped offset cells count once.
    Inputs must already be wrapped into the domain, as in the production search.
    """
    counts = grid.numCells.to(torch.int64)
    shape = tuple(int(c) for c in counts.cpu())
    stride = torch.tensor([math.prod(shape[:d]) for d in range(len(shape))], device=positions.device)
    table = grid.sortedCellTable
    occupancy = torch.zeros(math.prod(shape), dtype=torch.int64, device=positions.device)
    live = table[:, 2] > 0
    occupancy.scatter_add_(0, table[live, 0], table[live, 2])
    cells = torch.floor((positions - grid.qMin) / grid.hCell).to(torch.int64)
    flags = [bool(x) for x in periodic.cpu()]
    for d, flag in enumerate(flags):
        cells[:, d] = cells[:, d].remainder(shape[d]) if flag else cells[:, d].clamp(0, shape[d] - 1)
    offsets = {tuple(int(o[d]) % shape[d] if flags[d] else int(o[d]) for d in range(len(shape)))
               for o in grid.cellOffsets.cpu().tolist()}
    visits = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)
    for offset in sorted(offsets):
        other = cells + torch.tensor(offset, device=positions.device)
        valid = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        for d, flag in enumerate(flags):
            if flag:
                other[:, d].remainder_(shape[d])
            else:
                valid &= (other[:, d] >= 0) & (other[:, d] < shape[d])
                other[:, d].clamp_(0, shape[d] - 1)
        visits += torch.where(valid, occupancy[(other * stride).sum(1)], 0)
    return visits


def measure(call, repeat, warmup, batch=1):
    # Callables retain no large results: each allocation is released before the
    # next call, rather than holding two multi-GB CSR lists during assignment.
    for _ in range(warmup):
        call()
    torch.cuda.current_stream().synchronize()
    walls, intervals = [], []
    for _ in range(repeat):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        call()
        stop.record()
        stop.synchronize()
        walls.append((time.perf_counter() - before) * 1e3 / batch)
        intervals.append(start.elapsed_time(stop) / batch)
    return dict(wall_ms=walls, median_wall_ms=statistics.median(walls),
                device_interval_ms=intervals, median_device_interval_ms=statistics.median(intervals))


def graphMeasure(call, repeat, warmup, batch):
    from warpSPHCore.autograd.stream import torchWarpCapture, torchWarpStream
    eager = measure(call, repeat, warmup)
    graph = torch.cuda.CUDAGraph()
    before = time.perf_counter()
    with torchWarpStream(torch.device('cuda')), torchWarpCapture(graph):
        for _ in range(batch):
            call()
    torch.cuda.current_stream().synchronize()
    captureMs = (time.perf_counter() - before) * 1e3
    measured = measure(graph.replay, repeat, warmup, batch)
    return dict(eager=eager, graph=measured, batch=batch, capture_ms=captureMs)


def diagnosticStages(call, repeats=3):
    """Separate instrumented observations, never used for headline throughput."""
    pending = []

    @contextmanager
    def scope(name):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        try:
            yield
        finally:
            wall = (time.perf_counter() - before) * 1e3
            stop.record()
            pending.append((name, start, stop, wall))

    modules = [importlib.import_module('warpSPHCore.radiusSearch.compactHash.' + name)
               for name in ('wrapper', 'buildHashmap', 'search')]
    originals = [module.record_function for module in modules]
    try:
        for module in modules:
            module.record_function = scope
        for _ in range(repeats):
            call()
        torch.cuda.current_stream().synchronize()
    finally:
        for module, original in zip(modules, originals):
            module.record_function = original
    result = {}
    for name, start, stop, wall in pending:
        values = result.setdefault(name, dict(host_scope_ms=[], device_interval_ms=[]))
        values['host_scope_ms'].append(wall)
        values['device_interval_ms'].append(start.elapsed_time(stop))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', nargs='+', type=int, default=[1024, 65536, 262144, 1048576, 4194304])
    parser.add_argument('--neighbors', nargs='+', type=int, default=[55, 220])
    parser.add_argument('--layouts', nargs='+', choices=['lattice', 'jittered-shuffled'], default=['lattice', 'jittered-shuffled'])
    parser.add_argument('--dim', type=int, choices=[2, 3], default=2)
    parser.add_argument('--repeat', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--operator', action='store_true', help='also compare serial/32-lane Laplacian through this grid and CSR')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.ns, *args.neighbors, args.repeat, args.warmup, args.batch) <= 0:
        parser.error('positive sizes and sample counts required')
    if not torch.cuda.is_available():
        parser.error('accessible CUDA required')
    os.environ['WARPSPHCORE_PROFILING'] = '0'
    os.environ['WARPSPHCORE_AUDIT'] = '0'
    sys.path.insert(0, str(ROOT / 'src'))
    from warpSPHCore import (ParticleState, SupportScheme, radiusSearchCompactHashMap,
                             OperationProperties, KernelFunctions, WarpOperation,
                             GradientScheme, LaplacianScheme, warpOperation, setNeighborLanes)
    from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap, makeHashMapSlots, buildFixed, GraphedBuild
    from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.wp_countNeighbors import radiusSearchCountNeighborsCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors import radiusSearchCollectCompactHashMap
    from warpSPHCore.type_config import get_torch_precision, get_torch_index_dtype, get_torch_offset_dtype
    from warpSPHCore.util import generateNeighborTestData
    from warpSPHCore.math import getNextPrime

    torch.set_num_threads(2)
    wp.init()
    paths = sorted((ROOT / 'src/warpSPHCore').rglob('*.py')) + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(configuration=vars(args) | {'output': str(args.output)}, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, warp=wp.__version__, dtype=str(get_torch_precision()),
                  index_dtype=str(get_torch_index_dtype()), offset_dtype=str(get_torch_offset_dtype()),
                  source_sha256=hashes, records=[], complete=False, scope=__doc__,
                  gpu_free_at_start_bytes=torch.cuda.mem_get_info()[0], seed=17,
                  input_scope='fixed positions; warmed allocator; no temporal reuse/Verlet skin; no production code changes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    caseIndex = 0
    for requested in args.neighbors:
        for n in args.ns:
            side = round(n ** (1 / args.dim))
            if side ** args.dim != n:
                parser.error('N must be an exact dim-th power')
            for layout in args.layouts:
                torch.cuda.reset_peak_memory_stats()
                free = torch.cuda.mem_get_info()[0]
                positions, supports, _, domain, dx = generateNeighborTestData(side, requested, args.dim, True, 'cuda')
                if positions.shape[0] != n:
                    raise RuntimeError('unexpected generated size')
                if layout == 'jittered-shuffled':
                    rng = torch.Generator(device='cuda').manual_seed(17)
                    positions += (torch.rand(positions.shape, device='cuda', dtype=positions.dtype, generator=rng) - .5) * (.4 * dx)
                    positions = (positions - domain.min).remainder(domain.max - domain.min) + domain.min
                    order = torch.randperm(n, device='cuda', generator=rng)
                    positions, supports = positions[order].contiguous(), supports[order].contiguous()
                    del order
                particles = ParticleState(positions=positions, supports=supports,
                                          masses=torch.full_like(supports, dx ** args.dim), densities=torch.ones_like(supports),
                                          kinds=torch.zeros(n, dtype=torch.int32, device='cuda'))
                H = getNextPrime(n)
                mode = SupportScheme.Gather
                periodic = domain.periodic
                grid = buildCompactHashMap(positions, positions, supports, supports, periodic, domain, mode, H)
                adjacency = radiusSearchOnCompactHashMap(grid, positions, positions, supports, supports, periodic, domain, mode, H)
                edges = adjacency.numEdges
                referenceCounts = adjacency.numNeighbors.clone()
                candidate = candidateVisits(positions, grid, periodic)
                if not bool((candidate >= referenceCounts).all()):
                    raise RuntimeError('candidate count smaller than accepted count')
                candidates = int(candidate.sum())
                # Independent distance oracle on every pair for small smoke cases.
                if n <= 4096:
                    expected = torch.zeros(n, dtype=torch.int64, device='cuda')
                    for begin in range(0, n, 256):
                        delta = (positions[begin:begin+256, None] - positions[None]).abs()
                        delta = torch.minimum(delta, domain.max - domain.min - delta)
                        expected[begin:begin+256] = (delta.square().sum(2) <= supports[begin:begin+256, None].square()).sum(1)
                    torch.testing.assert_close(referenceCounts.long(), expected, rtol=0, atol=0)
                    del expected, delta
                wpStream = wp.stream_from_torch(torch.cuda.current_stream())
                # Production wraps positions before these launches. All benchmark
                # positions already lie in the domain; use the same arithmetic.
                wrapped = (positions - domain.min).remainder(domain.max - domain.min) + domain.min
                counts = torch.empty_like(referenceCounts)
                prefix = torch.empty_like(adjacency.edgeOffsets)
                prefix[0] = 0
                common = [wp.from_torch(x) if isinstance(x, torch.Tensor) else x for x in
                          [wrapped, supports, grid.sortedPositions, grid.sortedSupports, grid.hashTable, grid.sortedCellTable,
                           grid.qMin, grid.hCell, grid.D, grid.cellOffsets, grid.numCells, grid.qMax, grid.qMin, periodic, wp.uint32(mode.value)]]
                def count():
                    wp.launch(radiusSearchCountNeighborsCompactHashMap, n, inputs=common + [wp.from_torch(counts)], stream=wpStream)
                def scan():
                    # The production build above has already checked E fits.
                    torch.cumsum(counts[:-1], 0, dtype=prefix.dtype, out=prefix[1:])
                def collect():
                    wp.launch(radiusSearchCollectCompactHashMap, n, inputs=common + [wp.from_torch(counts), wp.from_torch(prefix),
                              wp.from_torch(grid.sortIndex), wp.from_torch(adjacency.paddedI), wp.from_torch(adjacency.paddedJ)], stream=wpStream)
                sample = torch.linspace(0, edges - 1, min(edges, 4096), device='cuda').long()
                sampleI, sampleJ = adjacency.paddedI[sample].clone(), adjacency.paddedJ[sample].clone()
                bare = {}
                count()
                scan()
                # Rotate the independent timing order to reduce systematic drift.
                phases = [('count', count), ('prefix', scan), ('collect', collect)]
                order = caseIndex % len(phases)
                for name, call in phases[order:] + phases[:order]:
                    bare[name] = graphMeasure(call, args.repeat, args.warmup, args.batch)
                torch.testing.assert_close(counts, referenceCounts, rtol=0, atol=0)
                torch.testing.assert_close(prefix, adjacency.edgeOffsets, rtol=0, atol=0)
                torch.testing.assert_close(adjacency.paddedI[sample], sampleI, rtol=0, atol=0)
                torch.testing.assert_close(adjacency.paddedJ[sample], sampleJ, rtol=0, atol=0)
                row = dict(n=n, dim=args.dim, requested_neighbors=requested, layout=layout, edges=edges, degree=edges/n,
                           physical_nonself_pairs=edges-n, candidate_visits_per_grid_pass=candidates,
                           candidate_to_accepted_ratio=candidates/edges, bare=bare, hash_length=H,
                           occupied_cells=int((grid.sortedCellTable[:, 2] > 0).sum()), cell_offsets=grid.numOffsets,
                           cell_counts=grid.numCells.cpu().tolist(), h_cell=grid.hCell,
                           count_range=[int(referenceCounts.min()), int(referenceCounts.max())],
                           checked=dict(counts=True, prefix=True, sampled_pair_entries=sample.numel(), independent_small_distance_oracle=n<=4096),
                           free_before_case_bytes=free)
                if args.operator:
                    from bench_krylov import timeMatvec
                    field = torch.sin(positions[:, 0] * 3).contiguous()
                    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Laplacian,
                                                gradientMode=GradientScheme.Difference, laplacianMode=LaplacianScheme.Brookshaw,
                                                supportMode=mode)
                    row['laplacian'] = []
                    reference = None
                    for topologyName, topology in [('adjacency', adjacency), ('grid', grid)]:
                        for lanes in (1, 32):
                            setNeighborLanes(lanes)
                            def action(_):
                                return warpOperation(particles, props, domain, adjacency=topology,
                                                     queryValues=field, referenceValues=field)
                            result = action(field)
                            if reference is None:
                                reference = result.clone()
                            tolerance = 3e-12 if field.dtype == torch.float64 else 3e-5
                            torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance*max(1., float(reference.abs().max())))
                            timing = timeMatvec(action, field, repeat=args.repeat, batch=args.batch)
                            row['laplacian'].append(dict(traversal=topologyName, lanes=lanes, timing=timing,
                                relative_max_difference=float((result-reference).abs().max())/max(1., float(reference.abs().max()))))
                    setNeighborLanes(None)
                    del action, topology, field, result, reference
                del adjacency, sample, sampleI, sampleJ, candidate, common, counts, prefix, wrapped, count, scan, collect, phases, call
                gc.collect()
                torch.cuda.current_stream().synchronize()
                torch.cuda.empty_cache()

                def build():
                    buildCompactHashMap(positions, positions, supports, supports, periodic, domain, mode, H)
                def search():
                    radiusSearchOnCompactHashMap(grid, positions, positions, supports, supports, periodic, domain, mode, H)
                def full():
                    radiusSearchCompactHashMap(particles, domain, mode=mode)
                calls = [('grid_build', build), ('search_prebuilt_grid', search), ('full_rebuild', full)]
                rotation = caseIndex % len(calls)
                row['public'] = {}
                for name, call in calls[rotation:] + calls[:rotation]:
                    row['public'][name] = measure(call, args.repeat, args.warmup)
                row['diagnostic_stage_intervals'] = diagnosticStages(full)
                # Fixed-shape build baseline, same radius/grid/hash length. Copy
                # inputs on each call, and include that cost in both variants.
                slots = makeHashMapSlots(n, domain, float(supports.max()), device='cuda', mode=mode, hashMapLength=H)
                graphed = GraphedBuild(slots)
                def fixed():
                    slots.positions.copy_(positions)
                    slots.supports.copy_(supports)
                    buildFixed(slots)
                def cached():
                    slots.positions.copy_(positions)
                    slots.supports.copy_(supports)
                    graphed.replay()
                before = time.perf_counter()
                cached()
                cached()
                torch.cuda.current_stream().synchronize()
                row['fixed_build_setup_and_capture_ms'] = (time.perf_counter() - before) * 1e3
                row['public']['fixed_grid_build'] = measure(fixed, args.repeat, args.warmup)
                row['public']['cached_grid_build'] = measure(cached, args.repeat, args.warmup)
                got = radiusSearchOnCompactHashMap(slots.hashMap, positions, positions, supports, supports, periodic, domain, mode, H)
                torch.testing.assert_close(got.numNeighbors, referenceCounts, rtol=0, atol=0)
                if got.numEdges != edges or not bool(slots.fits):
                    raise RuntimeError('fixed build does not match baseline')
                row['checked']['fixed_grid_counts_and_fits'] = True
                row['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
                # Search traverses the candidate cells twice, including self;
                # useful output pairs and distance-test visits remain distinct.
                row['rates'] = {}
                for name, timing in row['public'].items():
                    seconds = timing['median_wall_ms'] / 1000
                    row['rates'][name] = dict(particles_per_second=n/seconds,
                        physical_nonself_pairs_per_second=(edges-n)/seconds if 'rebuild' in name or 'search' in name else None,
                        candidate_tests_per_second=2*candidates/seconds if 'rebuild' in name or 'search' in name else None)
                report['records'].append(row)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(f'{args.dim}D/{layout} N={n} degree={edges/n:.2f} candidates/accepted={candidates/edges:.2f}: '
                      f'build {row["public"]["grid_build"]["median_wall_ms"]:.3f}, '
                      f'search {row["public"]["search_prebuilt_grid"]["median_wall_ms"]:.3f}, '
                      f'full {row["public"]["full_rebuild"]["median_wall_ms"]:.3f}, '
                      f'cached-build {row["public"]["cached_grid_build"]["median_wall_ms"]:.3f} ms', flush=True)
                del got, fixed, cached, slots, graphed, build, search, full, calls, call, grid, particles, positions, supports, referenceCounts
                gc.collect()
                torch.cuda.current_stream().synchronize()
                torch.cuda.empty_cache()
                caseIndex += 1
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError('measured source changed during sweep')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            main()
        stream.synchronize()
    else:
        main()
