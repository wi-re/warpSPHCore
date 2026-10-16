#!/usr/bin/env python3
"""Cross input/spatial query scheduling with fused/split collection, retaining CSR order."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bench_neighbor_search import measure, graphMeasure, candidateVisits


def cpuMeasure(call, repeat, warmup):
    for _ in range(warmup): call()
    values = []
    for _ in range(repeat):
        start = time.perf_counter(); call(); values.append((time.perf_counter() - start) * 1000)
    return dict(wall_ms=values, median_wall_ms=statistics.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=2)
    parser.add_argument('--ns', nargs='+', type=int, default=[1024, 65536, 262144, 1048576, 4194304])
    parser.add_argument('--neighbors', nargs='+', type=int, default=[55, 220])
    parser.add_argument('--layouts', nargs='+', choices=['lattice', 'jittered-shuffled'], default=['lattice', 'jittered-shuffled'])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.ns, *args.neighbors, args.repeat, args.warmup, args.batch) <= 0: parser.error('positive sizes/counts required')
    if args.device == 'cuda' and not torch.cuda.is_available(): parser.error('accessible CUDA required')
    from warpSPHCore import SupportScheme
    from warpSPHCore.radiusSearch.compactHash import buildCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.search import radiusSearchOnCompactHashMap
    from warpSPHCore.radiusSearch.compactHash.wp_countNeighbors import radiusSearchCountNeighborsCompactHashMap, radiusSearchCountNeighborsSpatial
    from warpSPHCore.radiusSearch.compactHash.wp_collectNeighbors import (radiusSearchCollectCompactHashMap,
        radiusSearchCollectJOnly, radiusSearchCollectSpatial, radiusSearchCollectJOnlySpatial, fillCSRRows)
    from warpSPHCore.type_config import get_torch_precision, get_torch_index_dtype, get_torch_offset_dtype
    from warpSPHCore.util import generateNeighborTestData
    from warpSPHCore.math import getNextPrime

    torch.set_num_threads(2); wp.init()
    paths = sorted((ROOT / 'src/warpSPHCore').rglob('*.py')) + [Path(__file__), Path(__file__).with_name('bench_neighbor_search.py')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(complete=False, configuration=vars(args) | {'output': str(args.output)}, records=[],
        dtype=str(get_torch_precision()), index_dtype=str(get_torch_index_dtype()), offset_dtype=str(get_torch_offset_dtype()),
        gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None, torch=torch.__version__, warp=wp.__version__,
        source_sha256=hashes, scope=__doc__, seed=17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampleTime = measure if args.device == 'cuda' else cpuMeasure
    stream = wp.stream_from_torch(torch.cuda.current_stream()) if args.device == 'cuda' else None
    kernels = {'input_fused': radiusSearchCollectCompactHashMap, 'input_split': radiusSearchCollectJOnly,
               'spatial_fused': radiusSearchCollectSpatial, 'spatial_split': radiusSearchCollectJOnlySpatial}
    case = 0
    for requested in args.neighbors:
        for n in args.ns:
            side = round(n ** (1 / args.dim))
            if side ** args.dim != n: parser.error('N must be an exact dim-th power')
            for layout in args.layouts:
                p, h, _, domain, dx = generateNeighborTestData(side, requested, args.dim, True, args.device)
                if layout == 'jittered-shuffled':
                    rng = torch.Generator(device=args.device).manual_seed(17)
                    p += (torch.rand(p.shape, dtype=p.dtype, device=p.device, generator=rng) - .5) * (.4 * dx)
                    p = (p - domain.min).remainder(domain.max - domain.min) + domain.min
                    perm = torch.randperm(n, device=p.device, generator=rng)
                    p, h = p[perm].contiguous(), h[perm].contiguous(); del perm
                H = getNextPrime(n); mode = SupportScheme.Gather
                grid = buildCompactHashMap(p, p, h, h, domain.periodic, domain, mode, H)
                def search(order='input', collection='fused'):
                    return radiusSearchOnCompactHashMap(grid, p, p, h, h, domain.periodic, domain, mode, H,
                                                        queryOrder=order, collection=collection)
                reference = search(); edges = reference.numEdges
                counts, offsets = reference.numNeighbors.clone(), reference.edgeOffsets.clone()
                candidateCount = int(candidateVisits(p, grid, domain.periodic).sum())
                sample = torch.linspace(0, edges - 1, min(edges, 65536), device=p.device).long()
                expectedJ = reference.paddedJ[sample].clone()
                wrapped = (p - domain.min).remainder(domain.max - domain.min) + domain.min
                common = [wp.from_torch(x) if isinstance(x, torch.Tensor) else x for x in
                    [wrapped, h, grid.sortedPositions, grid.sortedSupports, grid.hashTable, grid.sortedCellTable,
                     grid.qMin, grid.hCell, grid.D, grid.cellOffsets, grid.numCells, grid.qMax, grid.qMin,
                     domain.periodic, wp.uint32(mode.value)]]
                countInputs = common + [wp.from_torch(reference.numNeighbors)]
                collectInputs = common + [wp.from_torch(counts), wp.from_torch(offsets), wp.from_torch(grid.sortIndex),
                                         wp.from_torch(reference.paddedI), wp.from_torch(reference.paddedJ)]
                bare = {}; variants = list(kernels); rotation = case % 4
                for order, kernel in [('input', radiusSearchCountNeighborsCompactHashMap), ('spatial', radiusSearchCountNeighborsSpatial)]:
                    inputs = countInputs + ([wp.from_torch(grid.sortIndex)] if order == 'spatial' else [])
                    def count(): wp.launch(kernel, n, inputs=inputs, device=args.device, stream=stream)
                    bare['count_' + order] = graphMeasure(count, args.repeat, args.warmup, args.batch) if stream else cpuMeasure(count, args.repeat, args.warmup)
                    torch.testing.assert_close(reference.numNeighbors, counts, rtol=0, atol=0)
                for name in variants[rotation:] + variants[:rotation]:
                    def collect():
                        wp.launch(kernels[name], n, inputs=collectInputs, device=args.device, stream=stream)
                        if name.endswith('split'):
                            wp.launch(fillCSRRows, (n, 32), inputs=[collectInputs[15], collectInputs[16], collectInputs[18]],
                                      block_dim=128, device=args.device, stream=stream)
                    bare[name] = graphMeasure(collect, args.repeat, args.warmup, args.batch) if stream else cpuMeasure(collect, args.repeat, args.warmup)
                    torch.testing.assert_close(reference.paddedJ[sample], expectedJ, rtol=0, atol=0)
                    for begin in range(0, n, 65536):
                        endRow = min(begin + 65536, n)
                        start, end = int(offsets[begin]), int(offsets[endRow - 1] + counts[endRow - 1])
                        ids = torch.arange(begin, endRow, dtype=get_torch_index_dtype(), device=p.device)
                        expectedI = torch.repeat_interleave(ids, counts[begin:endRow].long(), output_size=end - start)
                        torch.testing.assert_close(reference.paddedI[start:end], expectedI, rtol=0, atol=0)
                    assert bool((reference.paddedI[edges:] == -1).all()) and bool((reference.paddedJ[edges:] == -1).all())
                storage = dict(pair_bytes=(reference.paddedI.numel() + reference.paddedJ.numel()) * reference.paddedI.element_size(),
                               extra_query_sort_bytes=0, extra_query_sort_launches=0)
                del reference, common, countInputs, collectInputs, count, collect, inputs, wrapped, expectedI, ids
                gc.collect()
                if stream: torch.cuda.current_stream().synchronize(); torch.cuda.empty_cache()
                public = {}
                for name in variants[rotation:] + variants[:rotation]:
                    order, collection = name.split('_')
                    if stream: torch.cuda.reset_peak_memory_stats(); baseline = torch.cuda.memory_allocated()
                    public[name] = sampleTime(lambda: search(order, collection), args.repeat, args.warmup)
                    public[name]['useful_nonself_pairs_per_second'] = (edges - n) * 1000 / public[name]['median_wall_ms']
                    public[name]['candidate_distance_tests_per_second'] = 2 * candidateCount * 1000 / public[name]['median_wall_ms']
                    if stream: public[name]['peak_additional_torch_allocated_bytes'] = torch.cuda.max_memory_allocated() - baseline
                    got = search(order, collection)
                    torch.testing.assert_close(got.numNeighbors, counts, rtol=0, atol=0)
                    torch.testing.assert_close(got.edgeOffsets, offsets, rtol=0, atol=0)
                    torch.testing.assert_close(got.paddedJ[sample], expectedJ, rtol=0, atol=0)
                    del got
                row = dict(n=n, dim=args.dim, layout=layout, requested_neighbors=requested, edges=edges, degree=edges / n,
                    candidate_visits_per_grid_pass=candidateCount, public=public, bare=bare, storage=storage,
                    checks=dict(exact_counts_offsets=True, all_i_per_collect_variant=True, j_samples_per_variant=sample.numel(), padding=True))
                report['records'].append(row); args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(f'{args.dim}D {layout} N={n} k={edges/n:.2f}: '+str({k: round(v['median_wall_ms'], 3) for k,v in public.items()}), flush=True)
                del search, grid, p, h, counts, offsets, sample, expectedJ
                gc.collect()
                if stream: torch.cuda.current_stream().synchronize(); torch.cuda.empty_cache()
                case += 1
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in hashes.items())
    report['complete'] = True; args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    os.environ['WARPSPHCORE_PROFILING'] = '0'; os.environ['WARPSPHCORE_AUDIT'] = '0'
    # A side stream keeps capture separate from the legacy default stream.
    if '--device' not in sys.argv or sys.argv[sys.argv.index('--device') + 1] == 'cuda':
        with torch.cuda.stream(torch.cuda.Stream()): main()
    else: main()
