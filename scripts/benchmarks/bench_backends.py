#!/usr/bin/env python3
"""Compare fresh search and fixed-CSR primals across the three execution paths.

Compilation, fixture creation, CSR canonicalization and correctness checks are
outside timings. The optional CMake bridge keeps standalone iteration loops in
C++, with no Torch or Python calls inside the measured interval.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


class NativeCase(C.Structure):
    """Non-owning fixture descriptor; layout matches native/runner.cpp."""
    _fields_ = [
        *((name, C.c_int) for name in ('device', 'dimension', 'operation', 'threads', 'precision')),
        ('particles', C.c_int64), ('edges', C.c_int64), ('stream', C.c_void_p),
        *((name, C.c_void_p) for name in (
            'positions', 'supports', 'masses', 'densities', 'values', 'lower', 'upper',
            'periodic', 'offsets', 'indices', 'counts', 'kinds', 'output')),
    ]


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return dict(command=command, returncode=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, error=str(error))


def activity():
    snapshot = dict(timestamp=time.time(), gpu=command_output([
        'nvidia-smi', '--query-gpu=name,index,utilization.gpu,utilization.memory,memory.used,memory.total',
        '--format=csv']), processes=command_output([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
        cpu=command_output(['ps', '-eo', 'pid,pcpu,comm', '--sort=-pcpu']))
    if 'stdout' in snapshot['cpu']:
        snapshot['cpu']['stdout'] = '\n'.join(snapshot['cpu']['stdout'].splitlines()[:21])
    return snapshot


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def native_library(path):
    lib = C.CDLL(str(path.resolve()))
    if not hasattr(lib, "native_benchmark_abi_version") or lib.native_benchmark_abi_version() != 2:
        raise RuntimeError("Rebuild the timing bridge: benchmark descriptor ABI 2 is required")
    lib.native_benchmark.argtypes = [C.POINTER(NativeCase), C.c_int, C.c_int,
                                    C.POINTER(C.c_double), C.POINTER(C.c_double)]
    lib.native_benchmark.restype = C.c_int
    lib.native_benchmark_search.argtypes = [C.POINTER(NativeCase), C.c_void_p]
    lib.native_benchmark_search.restype = C.c_int
    lib.native_benchmark_error.restype = C.c_char_p
    return lib


def native_sample(lib, request, task, iterations):
    completed, submitted = C.c_double(), C.c_double()
    status = lib.native_benchmark(C.byref(request), task, iterations,
                                  C.byref(completed), C.byref(submitted))
    if status:
        raise RuntimeError(lib.native_benchmark_error().decode())
    return completed.value, submitted.value


def native_request(q, domain, values, adjacency, output, operation, threads):
    request = NativeCase()
    request.device = (2 if torch.version.hip else 1) if q.positions.is_cuda else 0
    request.dimension = domain.dim
    request.operation = operation.value
    request.threads = threads
    request.precision = 0 if q.positions.dtype == torch.float32 else 1
    request.particles = len(q.positions)
    request.edges = adjacency.numEdges
    if request.device:
        request.stream = torch.cuda.current_stream().cuda_stream
    buffers = dict(positions=q.positions, supports=q.supports, masses=q.masses,
                   densities=q.densities, values=values, lower=domain.min, upper=domain.max,
                   periodic=domain.periodic, offsets=adjacency.edgeOffsets,
                   indices=adjacency.paddedJ, counts=adjacency.numNeighbors,
                   kinds=q.kinds, output=output)
    for name, tensor in buffers.items():
        setattr(request, name, tensor.data_ptr())
    return request


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def python_sample(call, device, iterations):
    synchronize(device)
    begin = time.perf_counter()
    # Retain only the last output, matching a typical eager timestep loop.
    for _ in range(iterations):
        output = call()
    submitted = time.perf_counter()
    synchronize(device)
    completed = time.perf_counter()
    return (completed - begin) * 1000 / iterations, (submitted - begin) * 1000 / iterations


def measure(sample, warmup, repeat):
    for _ in range(warmup):
        sample()
    samples = [sample() for _ in range(repeat)]
    completed, submitted = zip(*samples)
    return dict(completed_ms=list(completed), submitted_ms=list(submitted),
                median_ms=statistics.median(completed), min_ms=min(completed), max_ms=max(completed),
                median_submitted_ms=statistics.median(submitted))


def fixture(n, dimension, device, neighbors):
    generator = torch.Generator().manual_seed(2207 + n * 3 + dimension)
    positions = torch.rand((n, dimension), generator=generator, dtype=torch.float64)
    # Unit periodic box, uniform random points, approximately constant degree.
    # The cap prevents excessive radius in tiny cases; actual degree is recorded.
    unit_ball = (2., math.pi, 4 * math.pi / 3)[dimension - 1]
    radius = min(.43, (neighbors / (n * unit_ball)) ** (1 / dimension))
    supports = radius * (.98 + .04 * torch.rand(n, generator=generator, dtype=torch.float64))
    values = torch.sin(positions * (2 * math.pi)) + .1 * positions
    q = sph.ParticleState(positions.to(device), supports.to(device),
                          torch.full((n,), 1 / n, dtype=torch.float64, device=device),
                          torch.zeros(n, dtype=torch.int32, device=device),
                          torch.ones(n, dtype=torch.float64, device=device))
    domain = sph.DomainDescription(q.positions.new_zeros(dimension), q.positions.new_ones(dimension),
                                    torch.ones(dimension, dtype=torch.bool, device=device), dimension)
    return q, domain, values.to(device)


def canonical(adjacency, n):
    counts = adjacency.numNeighbors
    rows = torch.repeat_interleave(torch.arange(n, device=counts.device, dtype=torch.int64),
                                   counts.long(), output_size=adjacency.numEdges)
    keys = torch.sort(rows * n + adjacency.paddedJ[:adjacency.numEdges].long()).values
    offsets = torch.cumsum(counts.long(), 0) - counts.long()
    return sph.AdjacencyList(None, keys.remainder(n), counts, offsets, n, n,
                              numEdges=adjacency.numEdges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    parser.add_argument('--dims', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument('--ns', nargs='+', type=int, default=[1, 16, 256, 4096, 32768, 131072, 262144, 524288, 1000000])
    parser.add_argument('--neighbors', type=int, default=32)
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    parser.add_argument('--threads', type=int, default=available_cpus,
                        help='CPU workers for native paths (default: CPUs available to this process)')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--operator-iterations', type=int, default=10)
    parser.add_argument('--search-iterations', type=int, default=1)
    parser.add_argument('--large-case-threshold', type=int, default=65536,
                        help='At/above this N use one operator call per batch, three repeats and one warmup')
    parser.add_argument('--native-library', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if (min(args.ns) < 1 or max(args.ns) > 1_000_000 or args.repeat < 2 or args.warmup < 1
            or min(args.threads, args.neighbors, args.operator_iterations, args.search_iterations, args.large_case_threshold) < 1):
        parser.error('positive sizes <= 1M, threads, iterations; repeat >= 2 and warmup >= 1 required')
    os.environ['WARPSPHCORE_BACKEND'] = 'warp'
    os.environ['warpSPHCore_PRECISION'] = 'float64'
    os.environ['warpSPHCore_INDEX_DTYPE'] = 'int64'
    os.environ['warpSPHCore_OFFSET_DTYPE'] = 'int64'
    global torch, sph
    import torch
    import warp as wp
    import warpSPHCore as sph
    from warpSPHCore.native.api import radiusSearch as native_search, sphOperation as native_operation
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    wp.init()
    # Only primals are timed. These options affect compilation, not primal work.
    operations = [sph.WarpOperation.Density, sph.WarpOperation.Interpolate, sph.WarpOperation.Gradient,
                  sph.WarpOperation.Divergence, sph.WarpOperation.Curl, sph.WarpOperation.Laplacian,
                  sph.WarpOperation.Covariance]
    for operation in operations:
        wp.set_module_options({'enable_backward': False},
                              module=f'warpSPHCore.coreOperations.native.wp_{operation.name.lower()}')
    lib = native_library(args.native_library)
    source_paths = sorted(path for path in (ROOT / 'src').rglob('*')
                          if path.suffix in {'.py', '.h', '.cuh', '.cpp', '.cu'} and path.is_file())
    source_paths += [Path(__file__).resolve(), Path(__file__).parent / 'native/runner.cpp',
                     ROOT / 'cmake/Standalone.cmake', Path(__file__).parent / 'native/CMakeLists.txt']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    build_root = next((parent for parent in args.native_library.resolve().parents
                       if (parent / 'CMakeCache.txt').exists()), None)
    build_files = [args.native_library.resolve()]
    if build_root:
        build_files += [(build_root / 'libwarpsph_core.so').resolve(), build_root / 'CMakeCache.txt']
    report = dict(schema_version=1, complete=False, source_sha256=hashes,
        build_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in build_files},
        configuration=vars(args) | {
        'native_library': str(args.native_library), 'output': str(args.output)},
        environment=dict(python=platform.python_version(), torch=torch.__version__, warp=wp.__version__,
                         platform=platform.platform(), cpu=command_output(['lscpu']),
                         git=command_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                         dirty=command_output(['git', '-C', str(ROOT), 'status', '--short']),
                         gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         native_build=os.environ.get('WARPSPHCORE_NATIVE_BUILD'),
                         torch_extensions_dir=os.environ.get('TORCH_EXTENSIONS_DIR')),
        scope=dict(precision='float64', csr='int64 offsets and indices; int32 counts, all paths', fields='vector; scalar density; matrix covariance',
                   search='fresh compact index, count, scan, allocate, collect; Gather; periodic minimum image',
                   warp_search='fused collection, input query order, no COO rows, no capacity headroom; N hash buckets',
                   operators='fixed identical sorted CSR; Wendland2, Difference, Brookshaw, TrueAllToToAll',
                   large_sampling='At/above large_case_threshold: one operator call/batch, at most three repeats, one warmup',
                   timing='completed wall time per call; compilation, fixture setup and correctness excluded',
                   submitted='host submission interval; includes internal waits, excludes final stream wait; not pure kernel time',
                   validation='normal execution only; debug content and CSR scans disabled in every timed path',
                   standalone='fresh output allocation each call, C++ loop',
                   allocation='Torch/Warp use Torch caching allocator; standalone fresh allocations use malloc/cudaMalloc/hipMalloc',
                   standalone_reuse='metadata only, reused output, C++ loop',
                   cpu_parallelism='Torch threads and standalone OpenMP rows explicitly configured; Warp CPU kernel execution differs',
                   caveat='small-N times include useful work: they bound calling overhead, they do not isolate it'),
        activity=[], records=[])
    save(args.output, report)
    for device in args.devices:
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        for dimension in args.dims:
            for n in args.ns:
                large = n >= args.large_case_threshold
                repeats = min(args.repeat, 3) if large else args.repeat
                warmup = 1 if large else args.warmup
                operator_iterations = 1 if large else args.operator_iterations
                report['activity'].append(activity())
                q, domain, values = fixture(n, dimension, device, args.neighbors)
                warp_search = lambda: sph.radiusSearchCompactHashMap(
                    q, domain, sph.SupportScheme.Gather, sph.HashMapLengthMode.Fixed, n,
                    capacity=None, collection='fused', queryOrder='input', storeI=False)
                torch_search = lambda: native_search(q, domain)
                # Qualify freshly constructed neighborhoods before any measurement.
                adjacency = canonical(torch_search(), n)
                warp_adjacency = canonical(warp_search(), n)
                torch.testing.assert_close(adjacency.numNeighbors, warp_adjacency.numNeighbors, rtol=0, atol=0)
                torch.testing.assert_close(adjacency.paddedJ, warp_adjacency.paddedJ, rtol=0, atol=0)
                shared = sph.AdjacencyList(None, adjacency.paddedJ.to(sph.get_torch_index_dtype()),
                    adjacency.numNeighbors, adjacency.edgeOffsets.to(sph.get_torch_offset_dtype()),
                    n, n, numEdges=adjacency.numEdges)
                base = dict(device=device, dimension=dimension, n=n, edges=adjacency.numEdges,
                            mean_neighbors=adjacency.numEdges/n, warmup_batches=warmup)
                dummy = q.positions.new_empty(n)
                request = native_request(q, domain, values, adjacency, dummy,
                                          sph.WarpOperation.Density, args.threads)
                collected = torch.empty_like(adjacency.paddedJ)
                if lib.native_benchmark_search(C.byref(request), collected.data_ptr()):
                    raise RuntimeError(lib.native_benchmark_error().decode())
                torch.testing.assert_close(collected, adjacency.paddedJ, rtol=0, atol=0)
                searches = [('warp', lambda: python_sample(warp_search, device, args.search_iterations)),
                            ('native_torch', lambda: python_sample(torch_search, device, args.search_iterations)),
                            ('standalone', lambda: native_sample(lib, request, 1, args.search_iterations))]
                for backend, sample in searches:
                    report['records'].append(base | dict(operation='search', backend=backend,
                        iterations=args.search_iterations, correctness='exact CSR agreement',
                        **measure(sample, warmup, repeats)))
                if large:
                    save(args.output, report)
                    print(f'{device} D={dimension} N={n}: search qualified and timed', flush=True)
                for operation in operations:
                    properties = sph.OperationProperties(sph.KernelFunctions.Wendland2, operation=operation,
                        gradientMode=sph.GradientScheme.Difference,
                        laplacianMode=sph.LaplacianScheme.Brookshaw,
                        operationMode=sph.OperationDirection.TrueAllToToAll)
                    warp_call = lambda: sph.warpOperation(q, properties, domain, values, adjacency=shared)
                    torch_call = lambda: native_operation(q, properties, domain, values, adjacency=shared)
                    expected, actual = warp_call(), torch_call()
                    torch.testing.assert_close(actual, expected, rtol=3e-9, atol=3e-9)
                    out = torch.empty_like(actual)
                    request = native_request(q, domain, values, adjacency, out, operation, args.threads)
                    native_sample(lib, request, 2, 1)
                    torch.testing.assert_close(out, actual, rtol=3e-9, atol=3e-9)
                    # Qualify normal standalone execution before timing.
                    samples = [('warp', lambda: python_sample(warp_call, device, operator_iterations)),
                               ('native_torch', lambda: python_sample(torch_call, device, operator_iterations))]
                    for backend, sample in samples:
                        report['records'].append(base | dict(operation=operation.name.lower(), backend=backend,
                            iterations=operator_iterations, correctness='qualified',
                            **measure(sample, warmup, repeats)))
                    for backend, task in [('standalone', 0), ('standalone_reuse', 2)]:
                        sample = lambda: native_sample(lib, request, task, operator_iterations)
                        report['records'].append(base | dict(operation=operation.name.lower(), backend=backend,
                            iterations=operator_iterations, correctness='qualified',
                            **measure(sample, warmup, repeats)))
                    if large:
                        save(args.output, report)
                        print(f'{device} D={dimension} N={n}: {operation.name} qualified and timed', flush=True)
                synchronize(device)
                save(args.output, report)
                print(f'{device} D={dimension} N={n}: {adjacency.numEdges} edges; all paths qualified and timed', flush=True)
    report['activity'].append(activity())
    report['complete'] = True
    save(args.output, report)
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
