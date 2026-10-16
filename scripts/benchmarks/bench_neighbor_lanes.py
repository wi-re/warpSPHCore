#!/usr/bin/env python3
"""Measure trial curl/Laplacian lane choices against the serial kernels.

Records wall/device samples, interaction ledgers and serial-output differences.
Builds the CSR/grid once, warms all measured calls, and rotates lane order per
case. No operator default is selected from the timing inside this script.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch
import warp as wp

from _work_metrics import WorkLedger
from bench_krylov import timeMatvec
from bench_sph_work import measure


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', nargs='+', type=int, default=[1024, 65536, 262144, 1048576, 4194304])
    parser.add_argument('--neighbors', nargs='+', type=int, default=[25, 55, 110, 220])
    parser.add_argument('--lanes', nargs='+', type=int, choices=[1, 2, 4, 8, 16, 32], default=[1, 4, 8, 16, 32])
    parser.add_argument('--traversals', nargs='+', choices=['adjacency', 'grid'], default=['adjacency'])
    parser.add_argument('--dim', type=int, choices=[2, 3], default=2)
    parser.add_argument('--repeat', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.ns, *args.neighbors, args.repeat, args.warmup, args.batch) <= 0:
        parser.error('sizes and sample counts must be positive')
    if not torch.cuda.is_available():
        parser.error('requires an accessible CUDA device')
    sys.path.insert(0, str(ROOT / 'src'))
    from warpSPHCore import OperationProperties, ParticleState, radiusSearchCompactHashMap, setNeighborLanes, warpOperation
    from warpSPHCore.coreOperations.wp_curl import _CURL_SPEC
    from warpSPHCore.coreOperations.wp_laplacian import _LAPLACIAN_SPEC
    from warpSPHCore.enumTypes import GradientScheme, KernelFunctions, LaplacianScheme, SupportScheme, WarpOperation
    from warpSPHCore.type_config import get_torch_precision
    from warpSPHCore.util import generateNeighborTestData

    paths = sorted((ROOT / 'src/warpSPHCore').rglob('*.py'))
    paths += [Path(__file__), Path(__file__).with_name('_work_metrics.py'), Path(__file__).with_name('bench_krylov.py'),
              Path(__file__).with_name('bench_sph_work.py')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(configuration=vars(args) | {'output': str(args.output)}, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, warp=wp.__version__, dtype=str(get_torch_precision()), source_sha256=hashes,
                  records=[], complete=False, scope=__doc__,
                  gpu_free_at_start_bytes=torch.cuda.mem_get_info()[0],
                  serial_defaults={'curl': _CURL_SPEC.defaultNeighborLanes, 'laplacian': _LAPLACIAN_SPEC.defaultNeighborLanes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    wp.init()
    caseIndex = 0
    for neighbors in args.neighbors:
        for n in args.ns:
            torch.cuda.reset_peak_memory_stats()
            freeBeforeCase = torch.cuda.mem_get_info()[0]
            side = round(n ** (1 / args.dim))
            if side ** args.dim != n:
                parser.error('N must be an exact dim-th power')
            positions, supports, _, domain, dx = generateNeighborTestData(side, neighbors, args.dim, True, 'cuda')
            if positions.shape[0] != n:
                raise RuntimeError('generated lattice size differs from requested N')
            g = torch.Generator(device='cuda').manual_seed(0)
            scalar = torch.randn(n, generator=g, device='cuda', dtype=positions.dtype)
            vector = torch.randn((n, args.dim), generator=g, device='cuda', dtype=positions.dtype)
            particles = ParticleState(positions=positions, supports=supports, masses=torch.full_like(scalar, dx ** args.dim),
                                      densities=torch.ones_like(scalar), kinds=torch.zeros(n, device='cuda', dtype=torch.int32))
            adjacency, grid = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather, returnCompactHashMap=True)
            edges = int(adjacency.numNeighbors.sum())
            for traversal in args.traversals:
                topology = adjacency if traversal == 'adjacency' else grid
                for name, operation, field in [('laplacian', WarpOperation.Laplacian, scalar), ('curl', WarpOperation.Curl, vector)]:
                    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=operation,
                                                gradientMode=GradientScheme.Difference, laplacianMode=LaplacianScheme.Brookshaw,
                                                supportMode=SupportScheme.Gather)
                    call = lambda: warpOperation(particles, props, domain, adjacency=topology, queryValues=field, referenceValues=field)
                    setNeighborLanes(1)
                    reference = call().clone()
                    torch.cuda.current_stream().synchronize()
                    order = args.lanes[caseIndex % len(args.lanes):] + args.lanes[:caseIndex % len(args.lanes)]
                    caseIndex += 1
                    for lane in order:
                        setNeighborLanes(lane)
                        result, timing = measure(call, repeat=args.repeat, warmup=args.warmup)
                        difference = float((result - reference).abs().max())
                        scale = max(float(reference.abs().max()), 1.)
                        tolerance = 3e-12 if result.dtype == torch.float64 else 3e-5
                        if not bool(torch.isfinite(result).all()) or difference > tolerance * scale:
                            raise RuntimeError(f'{name}, N={n}, degree={edges/n}, lanes={lane}: serial disagreement {difference / scale}')
                        isolated = timeMatvec(lambda _: call(), field, repeat=args.repeat, batch=args.batch)
                        ledger = WorkLedger().pair(name, evaluated_nonself_pairs=edges - n,
                                                   candidate_entries=edges if traversal == 'adjacency' else None,
                                                   basis='equal-support lattice physical pairs from exact CSR; grid candidate visits not inferred')
                        record = dict(n=n, dim=args.dim, operation=name, field_shape=list(field.shape[1:]),
                                      requested_neighbors=neighbors, degree=edges / n, edges=edges, traversal=traversal,
                                      lanes=lane, lane_order=order, max_serial_difference=difference,
                                      free_before_case_bytes=freeBeforeCase,
                                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      relative_max_serial_difference=difference / scale, **timing, isolated=isolated,
                                      work=ledger.report(wall_ms=timing['median_wall_ms'], timing_scope='public operator wall call'),
                                      device_work=ledger.report(wall_ms=isolated['median_graph_device_ms_per_call'],
                                                               timing_scope='per-call device interval in warmed graph batch'))
                        report['records'].append(record)
                        args.output.write_text(json.dumps(report, indent=2) + '\n')
                        print(f'{traversal}/{name} N={n} degree={edges/n:g} lanes={lane}: '
                              f'{timing["median_wall_ms"]:.4f} ms wall, '
                              f'{isolated["median_graph_device_ms_per_call"]:.4f} ms graph, '
                              f'{record["device_work"]["directed_interactions_per_second"]/1e9:.2f} Ginteractions/s', flush=True)
            # No graphs or input buffers are reused across geometries. Release
            # the previous topology before building the next large CSR list.
            del call, topology, adjacency, grid, particles, positions, supports, scalar, vector, field, reference, result
            torch.cuda.current_stream().synchronize()
            gc.collect()
            torch.cuda.empty_cache()
    setNeighborLanes(None)
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError('measured source changed during the sweep')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            main()
        stream.synchronize()
    else:
        main()
