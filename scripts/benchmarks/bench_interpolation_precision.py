#!/usr/bin/env python3
"""Compare FP32/FP64 interpolation using one independently qualified, fixed CSR.

Run --prepare once, then run each precision/device/worker configuration in a
fresh process. Fixture construction, casts, compilation and correctness are
outside the measured intervals. Standalone timed loops execute entirely in C++.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--n', type=int, default=1_000_000)
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.n < 1:
        parser.error('Positive particle and worker counts are required')
    if not args.prepare and (args.output is None or args.native_library is None):
        parser.error('Measurements require --output and --native-library')

    precision = 'float64' if args.prepare else args.precision
    os.environ.update(WARPSPHCORE_BACKEND='warp', warpSPHCore_PRECISION=precision,
                      warpSPHCore_INDEX_DTYPE='int64', warpSPHCore_OFFSET_DTYPE='int64')
    sys.path.insert(0, str(args.root / 'scripts/benchmarks'))
    import bench_backends as bench
    import torch
    import warp as wp
    import warpSPHCore as sph
    from warpSPHCore.native.api import radiusSearch, sphOperation
    bench.torch, bench.sph = torch, sph
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    wp.init()
    wp.set_module_options({'enable_backward': False},
                          module='warpSPHCore.coreOperations.native.wp_interpolate')

    if args.prepare:
        q, domain, values = bench.fixture(args.n, 2, 'cpu', 32)
        adjacency = bench.canonical(radiusSearch(q, domain), args.n)
        warp_adjacency = bench.canonical(sph.radiusSearchCompactHashMap(
            q, domain, sph.SupportScheme.Gather, sph.HashMapLengthMode.Fixed, args.n,
            capacity=None, collection='fused', queryOrder='input', storeI=False), args.n)
        torch.testing.assert_close(adjacency.numNeighbors, warp_adjacency.numNeighbors, rtol=0, atol=0)
        torch.testing.assert_close(adjacency.paddedJ, warp_adjacency.paddedJ, rtol=0, atol=0)
        data = dict(positions=q.positions, supports=q.supports, masses=q.masses,
                    kinds=q.kinds, densities=q.densities, values=values,
                    lower=domain.min, upper=domain.max, periodic=domain.periodic,
                    counts=adjacency.numNeighbors, offsets=adjacency.edgeOffsets,
                    indices=adjacency.paddedJ, edges=adjacency.numEdges, n=args.n)
        torch.save(data, args.fixture)
        print(f'Prepared {args.n} particles, {adjacency.numEdges} edges; exact Warp/native CSR parity', flush=True)
        return

    data = torch.load(args.fixture, map_location='cpu', weights_only=True)
    n, edges = data['n'], data['edges']
    dtype = getattr(torch, args.precision)

    def make_inputs(scalar_type):
        def tensor(name):
            source = data[name]
            return source.to(device=args.device, dtype=scalar_type if source.is_floating_point() else source.dtype)
        particles = sph.ParticleState(*(tensor(name) for name in
                                        ['positions', 'supports', 'masses', 'kinds', 'densities']))
        domain = sph.DomainDescription(tensor('lower'), tensor('upper'), tensor('periodic'), 2)
        adjacency = sph.AdjacencyList(None, tensor('indices'), tensor('counts'), tensor('offsets'),
                                     n, n, numEdges=edges)
        return particles, domain, tensor('values'), adjacency

    q, domain, values, adjacency = make_inputs(dtype)
    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
        operation=sph.WarpOperation.Interpolate, gradientMode=sph.GradientScheme.Difference,
        laplacianMode=sph.LaplacianScheme.Brookshaw,
        operationMode=sph.OperationDirection.TrueAllToToAll)
    warp_call = lambda: sph.warpOperation(q, properties, domain, values, adjacency=adjacency)
    native_call = lambda: sphOperation(q, properties, domain, values, adjacency=adjacency)
    expected, actual = warp_call(), native_call()
    tolerance = 3e-5 if dtype == torch.float32 else 3e-9
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    out = torch.empty_like(actual)
    library = bench.native_library(args.native_library)
    request = bench.native_request(q, domain, values, adjacency, out,
                                    sph.WarpOperation.Interpolate, args.threads)
    bench.native_sample(library, request, 2, 1)
    torch.testing.assert_close(out, actual, rtol=tolerance, atol=tolerance)
    q64, domain64, values64, adjacency64 = make_inputs(torch.float64)
    reference64 = sphOperation(q64, properties, domain64, values64, adjacency=adjacency64)
    torch.testing.assert_close(actual.double(), reference64, rtol=tolerance, atol=tolerance)
    errors = dict(native_vs_warp_max_abs=(actual - expected).abs().max().item(),
                  standalone_vs_native_max_abs=(out - actual).abs().max().item(),
                  precision_vs_fp64_max_abs=(actual.double() - reference64).abs().max().item(),
                  rtol=tolerance, atol=tolerance)

    paths = [Path(__file__).resolve(), args.root / 'scripts/benchmarks/bench_backends.py',
             args.root / 'scripts/benchmarks/native/runner.cpp']
    paths += [path for path in (args.root / 'src').rglob('*')
              if path.is_file() and path.suffix in {'.py', '.h', '.cuh', '.cpp', '.cu'}]
    core_library = args.native_library.resolve().parents[3] / 'libwarpsph_core.so'
    iterations = 20 if args.device == 'cuda' else 1
    report = dict(complete=False, configuration=vars(args) | dict(root=str(args.root),
        fixture=str(args.fixture), native_library=str(args.native_library), output=str(args.output)),
        scope=dict(operation='interpolate', dimension=2, n=n, edges=edges,
                   mean_neighbors=edges/n, precision=args.precision, fixed_csr='Same FP64-qualified CSR for both precisions and devices',
                   timing='Completed wall time, resident inputs, search excluded',
                   warmup_batches=2, measured_batches=5, iterations=iterations),
        environment=dict(python=platform.python_version(), torch=torch.__version__, warp=wp.__version__,
                         gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         allowed_cpus=sorted(os.sched_getaffinity(0)),
                         omp={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES']},
                         cpu=bench.command_output(['lscpu'])),
        source_sha256={str(path.relative_to(args.root)):checksum(path) for path in paths},
        fixture_sha256=checksum(args.fixture),
        build_sha256={str(path):checksum(path) for path in [args.native_library, core_library]},
        correctness=errors, activity=[bench.activity()], records=[])
    bench.save(args.output, report)
    samples = [('warp', lambda: bench.python_sample(warp_call, args.device, iterations)),
               ('native_torch', lambda: bench.python_sample(native_call, args.device, iterations)),
               ('standalone', lambda: bench.native_sample(library, request, 0, iterations)),
               ('standalone_reuse', lambda: bench.native_sample(library, request, 2, iterations))]
    for backend, sample in samples:
        measurement = bench.measure(sample, 2, 5)
        report['records'].append(dict(backend=backend, **measurement))
        bench.save(args.output, report)
        print(f'{args.precision} {args.device} T={args.threads} {backend}: {measurement["median_ms"]:.4f} ms', flush=True)
    report['activity'].append(bench.activity())
    report['complete'] = True
    bench.save(args.output, report)


if __name__ == '__main__':
    main()
