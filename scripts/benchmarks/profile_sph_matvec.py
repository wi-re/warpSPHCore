#!/usr/bin/env python3
"""Separate instrumented SPH matvec pass; do not use as steady wall timing.

Summed kernel durations help distinguish small-N neighbor-chain latency from
Python launch gaps. This profiles ordinary kernels, avoiding conditional-loop
trace aggregation. Geometry and the action match bench_krylov's SPH workload.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import tempfile

import torch
import warp as wp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', type=int, default=32)
    parser.add_argument('--neighbors', type=int, default=55)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.side, args.neighbors, args.repeat) <= 0:
        parser.error('side, neighbors and repeat must be positive')
    if not torch.cuda.is_available():
        parser.error('requires an accessible CUDA device')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from warpSPHCore import OperationProperties, ParticleState, radiusSearchCompactHashMap, warpOperation
    from warpSPHCore.autograd.lanes import neighborLanes
    from warpSPHCore.coreOperations.wp_laplacian import _LAPLACIAN_SPEC
    from warpSPHCore.enumTypes import GradientScheme, KernelFunctions, LaplacianScheme, SupportScheme, WarpOperation
    from warpSPHCore.util import generateNeighborTestData

    wp.init()
    torch.set_num_threads(2)
    positions, supports, _, domain, dx = generateNeighborTestData(args.side, args.neighbors, 2, True, 'cuda')
    n = positions.shape[0]
    b = torch.randn(n, dtype=positions.dtype, device='cuda', generator=torch.Generator(device='cuda').manual_seed(0))
    particles = ParticleState(positions=positions, supports=supports, masses=torch.full_like(b, dx ** 2),
                              densities=torch.ones_like(b), kinds=torch.zeros(n, dtype=torch.int32, device=b.device))
    adjacency = radiusSearchCompactHashMap(particles, domain, mode=SupportScheme.Gather)
    props = OperationProperties(kernel=KernelFunctions.Wendland2, operation=WarpOperation.Laplacian,
                                gradientMode=GradientScheme.Difference, laplacianMode=LaplacianScheme.Brookshaw,
                                supportMode=SupportScheme.Gather)

    def matvec():
        lap = warpOperation(particles, props, domain, adjacency=adjacency, queryValues=b, referenceValues=b)
        return b - dx ** 2 * lap

    for _ in range(4):
        output = matvec()
    torch.cuda.current_stream().synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.repeat):
            output = matvec()
        torch.cuda.current_stream().synchronize()
    with tempfile.TemporaryDirectory(prefix='sph-profile-') as directory:
        trace = Path(directory) / 'trace.json'
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())['traceEvents']
    kernels = defaultdict(list)
    for event in events:
        if event.get('cat') == 'kernel':
            kernels[event['name']].append(event['dur'] / 1e3)
    report = dict(n=n, requested_neighbors=args.neighbors, edges=int(adjacency.numNeighbors.sum()),
                  actual_degree=int(adjacency.numNeighbors.sum()) / n,
                  repeat=args.repeat, configured_lanes=neighborLanes(),
                  laplacian_tiled=_LAPLACIAN_SPEC.tiledKernel is not None,
                  effective_laplacian_lanes=neighborLanes(default=_LAPLACIAN_SPEC.defaultNeighborLanes),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, warp=wp.__version__,
                  dtype=str(b.dtype), finite_output=bool(torch.isfinite(output).all()),
                  summed_kernel_ms_per_call=sum(sum(values) for values in kernels.values()) / args.repeat,
                  kernels=[dict(name=name, count=len(values), median_ms=statistics.median(values), durations_ms=values)
                           for name, values in kernels.items()],
                  scope='separate instrumented ordinary matvec; sums are kernel busy durations, not end-to-end wall time')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            main()
        stream.synchronize()
    else:
        main()
