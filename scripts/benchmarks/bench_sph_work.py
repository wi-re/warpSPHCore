#!/usr/bin/env python3
"""Operator interaction rates and CRK local-matrix work on fixed 2D lattices.

This is an operator/factor-stage benchmark, not a complete comp-SPH or CRK-SPH
time step. Counts are collected outside timing. Whole CRK includes volume,
moments, local coefficient construction and consistency density.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import warp as wp

from _work_metrics import WorkLedger
from bench_krylov import timeMatvec


def measure(call, *, repeat, warmup):
    for _ in range(warmup):
        result = call()
    torch.cuda.current_stream().synchronize()
    walls, intervals = [], []
    for _ in range(repeat):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter()
        result = call()
        stop.record()
        stop.synchronize()
        walls.append((time.perf_counter() - before) * 1e3)
        intervals.append(start.elapsed_time(stop))
    return result, dict(wall_ms=walls, median_wall_ms=statistics.median(walls),
                         device_interval_ms=intervals)


def main():
    import math

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns', nargs='+', type=int, default=[1024, 65536, 1048576])
    parser.add_argument('--neighbors', type=int, default=55)
    parser.add_argument('--operations', nargs='+',
                        choices=['density', 'interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance', 'crk'],
                        default=['density', 'interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance', 'crk'])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--matvec-batch', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.ns, args.neighbors, args.repeat, args.warmup, args.matvec_batch) <= 0:
        parser.error('sizes and sample counts must be positive')
    if not torch.cuda.is_available():
        parser.error('requires an accessible CUDA device')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'src'))
    from warpSPHCore import OperationProperties, ParticleState, radiusSearchCompactHashMap, warpOperation
    from warpSPHCore.autograd.lanes import neighborLanes
    from warpSPHCore.coreOperations.wp_curl import _CURL_SPEC
    from warpSPHCore.coreOperations.wp_laplacian import _LAPLACIAN_SPEC
    from warpSPHCore.crk import computeCRKFactors
    from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
    from warpSPHCore.crk.crk_terms import computeCRKTermsWarp
    from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend
    from warpSPHCore.enumTypes import GradientScheme, KernelFunctions, LaplacianScheme, SupportScheme, WarpOperation
    from warpSPHCore.util import generateNeighborTestData

    wp.init()
    torch.set_num_threads(2)
    rows = []
    for n in args.ns:
        side = math.isqrt(n)
        if side * side != n:
            parser.error('N must be a perfect square')
        positions, supports, _, domain, dx = generateNeighborTestData(side, args.neighbors, 2, True, 'cuda')
        if positions.shape[0] != n:
            raise RuntimeError('lattice size differs from requested N')
        scalar = torch.randn(n, dtype=positions.dtype, device='cuda', generator=torch.Generator(device='cuda').manual_seed(0))
        vector = torch.stack((scalar, scalar.roll(1)), dim=1).contiguous()
        particles = ParticleState(positions=positions, supports=supports, masses=torch.full_like(scalar, dx ** 2),
                                  densities=torch.ones_like(scalar), kinds=torch.zeros(n, dtype=torch.int32, device='cuda'))
        adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
        edges = int(adjacency.numNeighbors.sum())
        pairKw = dict(evaluated_nonself_pairs=edges - n, candidate_entries=edges,
                      basis='fixed all-to-all equal-support CSR without skin; self excluded from evaluated count')
        for name in args.operations:
            if name == 'crk':
                call = lambda: computeCRKFactors(particles, domain, KernelFunctions.Wendland2, adjacency=adjacency)
                output, timing = measure(call, repeat=args.repeat, warmup=args.warmup)
                area, density, state = output
                for tensor in (area, density, state.A, state.B, state.gradA, state.gradB):
                    if not bool(torch.isfinite(tensor).all()):
                        raise RuntimeError('nonfinite CRK factors')
                ledger = WorkLedger()
                for stage in ('CRK volume', 'CRK moments (fused outputs)', 'CRK consistency density'):
                    ledger.pair(stage, **pairKw)
                ledger.local_matrix('CRK determinant', matrices=n, rows=2, cols=2, method='torch.det')
                ledger.local_matrix('CRK pseudoinverse', matrices=n, rows=2, cols=2, method='torch.linalg.pinv (SVD)')
                ledger.particle('CRK coefficient contractions and limiter', particles=n)
                rows.append(dict(n=n, operation='crk_factors', edges=edges, degree=edges / n, **timing,
                                 work=ledger.report(wall_ms=timing['median_wall_ms'], timing_scope='whole CRK factor call, including consistency density')))
                # Isolate intra-particle work with frozen, production-generated
                # moments. This calls the same Torch routine as the wrapper.
                properties = OperationProperties(kernel=KernelFunctions.Wendland2, supportMode=SupportScheme.Gather)
                area = _computeCRKVolume_stateBackend(particles, properties, domain, adjacency=adjacency)
                properties = OperationProperties(kernel=KernelFunctions.Wendland2, supportMode=SupportScheme.Scatter)
                moments = _computeCRKMoments_stateBackend(particles, properties, domain, queryVolumes=area,
                                                         referenceVolumes=area, adjacency=adjacency)
                termCall = lambda: computeCRKTermsWarp(*moments[:6], num_nbrs=moments[6], supports=supports)
                terms, timing = measure(termCall, repeat=args.repeat, warmup=args.warmup)
                if not all(bool(torch.isfinite(t).all()) for t in terms):
                    raise RuntimeError('nonfinite isolated CRK terms')
                ledger = WorkLedger().local_matrix('CRK determinant', matrices=n, rows=2, cols=2, method='torch.det')
                ledger.local_matrix('CRK pseudoinverse', matrices=n, rows=2, cols=2, method='torch.linalg.pinv (SVD)')
                ledger.particle('CRK coefficient contractions and limiter', particles=n)
                rows.append(dict(n=n, operation='crk_local_terms', edges=edges, degree=edges / n, **timing,
                                 work=ledger.report(wall_ms=timing['median_wall_ms'], timing_scope='local CRK term assembly, frozen moments; determinant, SVD, contractions, limiter')))
            else:
                operation = getattr(WarpOperation, name.capitalize())
                properties = OperationProperties(kernel=KernelFunctions.Wendland2, operation=operation,
                                                  gradientMode=GradientScheme.Difference, laplacianMode=LaplacianScheme.Brookshaw,
                                                  supportMode=SupportScheme.Gather)
                values = vector if name in ('divergence', 'curl') else scalar
                call = lambda: warpOperation(particles, properties, domain, adjacency=adjacency,
                                              queryValues=values, referenceValues=values)
                output, timing = measure(call, repeat=args.repeat, warmup=args.warmup)
                if not bool(torch.isfinite(output).all()):
                    raise RuntimeError(f'nonfinite {name} result')
                isolated = timeMatvec(lambda _: call(), scalar, repeat=args.repeat, batch=args.matvec_batch)
                ledger = WorkLedger().pair(name, **pairKw)
                rows.append(dict(n=n, operation=name, edges=edges, degree=edges / n, **timing, isolated=isolated,
                                 work=ledger.report(wall_ms=timing['median_wall_ms'], timing_scope='public operator wall call'),
                                 device_work=ledger.report(wall_ms=isolated['median_graph_device_ms_per_call'],
                                                          timing_scope='warmed batch graph, device interval per public operator')))
            print(f'{name}, N={n}: {rows[-1]["median_wall_ms"]:.4f} ms', flush=True)
    paths = sorted((root / 'src/warpSPHCore').rglob('*.py')) + [Path(__file__), Path(__file__).with_name('_work_metrics.py'),
                                                           Path(__file__).with_name('bench_krylov.py')]
    report = dict(records=rows, gpu=torch.cuda.get_device_name(), torch=torch.__version__, warp=wp.__version__,
                  dtype=str(positions.dtype), configured_neighbor_lanes=neighborLanes(), requested_neighbors=args.neighbors,
                  operator_neighbor_lanes={'curl': neighborLanes(default=_CURL_SPEC.defaultNeighborLanes),
                                           'laplacian': neighborLanes(default=_LAPLACIAN_SPEC.defaultNeighborLanes)},
                  repeat=args.repeat, warmup=args.warmup, command=sys.argv,
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope=__doc__, complete=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            main()
        stream.synchronize()
    else:
        main()
