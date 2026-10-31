#!/usr/bin/env python3
"""Hardware-counter driver for the existing Morton interpolation experiments.

Search, ordering, planning and numerical verification happen before profiling.
Each marked range contains one complete interpolation call, including packing.
Use --counters under ncu --profile-from-start off --nvtx. Event timings must be
collected in a separate, uninstrumented process by omitting --counters.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--native-artifact', type=Path)
parser.add_argument('--precision', choices=['float32', 'float64'], required=True)
parser.add_argument('--dims', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--n-h', type=float, nargs='+', default=[4., 2.57])
parser.add_argument('--n', type=int, default=1000000)
parser.add_argument('--counters', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp', warpSPHCore_PRECISION=args.precision,
                  warpSPHCore_INDEX_DTYPE='int64', warpSPHCore_OFFSET_DTYPE='int64')
sys.path[:0] = [str(args.root/'src'), str(args.root/'scripts/benchmarks'),
               str(args.root/'experiments/cluster_pairs')]
import torch
import warp as wp
import warpSPHCore as sph
import warpSPHCore.native._build as build
from warpSPHCore.native.api import radiusSearch, sphOperation
from runtime import Library, build_cells, build_clusters, call, cell
import bench_backends as bench

torch.set_num_threads(16)
torch.set_num_interop_threads(1)
wp.init()
wp.set_module_options({'enable_backward': False}, module='warpSPHCore.coreOperations.native.wp_interpolate')
native_artifact = args.native_artifact or args.archive/'artifacts/optimized-native-torch.so'
cluster_artifact = args.archive/'artifacts/clusters-fixed.so'
torch.ops.load_library(str(native_artifact))
build._extension = torch.ops.warpsph_native
library = Library(cluster_artifact)
selection = json.loads((args.archive/f'screen-{args.precision}.json').read_text())['selection']
cell_selection = json.loads((args.archive/f'cell-screen-{args.precision}.json').read_text())['selection']
properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
    operation=sph.WarpOperation.Interpolate, operationMode=sph.OperationDirection.TrueAllToToAll)
report = dict(complete=False, pid=os.getpid(), counters=args.counters, cases=[], activity=[],
    environment=dict(torch=torch.__version__, warp=wp.__version__, gpu=str(torch.cuda.get_device_properties(0))),
    artifacts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [native_artifact, cluster_artifact]})

def sample(function):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(5):
        result = function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end)/5

for dimension in args.dims:
    for nh in args.n_h:
        report['activity'].append(bench.activity())
        n = args.n
        generator = torch.Generator().manual_seed(9067+n*3+dimension)
        x = torch.rand((n, dimension), generator=generator, dtype=torch.float64)
        side = round(n**(1/dimension))
        spacing = 1/side if side**dimension == n else n**(-1/dimension)
        x = x.to(device='cuda', dtype=getattr(torch, args.precision))
        h = x.new_full((n,), nh*spacing)
        state = sph.ParticleState(x, h, h.new_full((n,), 1/n),
            torch.zeros(n, device='cuda', dtype=torch.int32), h.new_ones(n))
        values = torch.sin(x*(2*math.pi))+.1*x
        domain = sph.DomainDescription(x.new_zeros(dimension), x.new_ones(dimension),
            torch.ones(dimension, device='cuda', dtype=torch.bool), dimension)
        cells = build_cells(library, state, values, float(h[0]), refinement=8)
        adjacency = radiusSearch(cells.state, domain)
        functions = {
            'native_csr': lambda: sphOperation(cells.state, properties, domain, cells.values, adjacency=adjacency),
            'warp_csr': lambda: sph.warpOperation(cells.state, properties, domain, cells.values, adjacency=adjacency)}
        key = f'{args.precision}:{dimension}:{nh:g}'
        block, width = cell_selection[key]['shared']
        tasks = cells.tasks(block//width)
        functions['cell_shared'] = lambda: cell.call(library, cells, tasks, 6, block, width)
        plans, configurations = {}, {}
        for name, compressed in [('cluster_raw', False), ('cluster_compressed', True)]:
            config = selection[key][f"cluster_{'compressed' if compressed else 'raw'}_super_shared"]
            tile = tuple(config[:4])
            if tile not in plans:
                plans[tile] = build_clusters(library, cells, *tile)
            plan = plans[tile]
            mode, compressed, warps = config[4:]
            functions[name] = lambda p=plan, m=mode, c=compressed, w=warps: call(library, p, m, c, w)
            configurations[name] = config
        expected = functions['native_csr']()
        tolerance = 3e-4 if args.precision == 'float32' else 3e-9
        parity = {}
        for name, function in functions.items():
            output = function()
            parity[name] = bool(torch.allclose(output, expected, rtol=tolerance, atol=tolerance))
            if not parity[name] and not (dimension == 1 and args.precision == 'float32' and name == 'warp_csr'):
                raise AssertionError(f'Unexpected parity failure: {key} {name}')
            for _ in range(4):
                function()
        torch.cuda.synchronize()
        record = dict(precision=args.precision, dimension=dimension, n_h=nh, n=n,
            neighbors=adjacency.numEdges/n, edges=adjacency.numEdges, parity=parity,
            configurations=configurations, cell_schedule=[6, block, width], timings={})
        if args.counters:
            for name, function in functions.items():
                label = f'{args.precision}_d{dimension}_h{nh:g}_{name}'
                torch.cuda.nvtx.range_push(label)
                torch.cuda.cudart().cudaProfilerStart()
                result = function()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                torch.cuda.nvtx.range_pop()
        else:
            samples = {name: [] for name in functions}
            for repeat in range(9):
                names = list(functions)
                random.Random(813+repeat).shuffle(names)
                for name in names:
                    samples[name].append(sample(functions[name]))
            record['timings'] = {name: dict(event_ms=statistics.median(data), samples_ms=data)
                                 for name, data in samples.items()}
        report['cases'].append(record)
        report['activity'].append(bench.activity())
        bench.save(args.output, report)
        print(f'Completed {key}', flush=True)
        # Drop owners before the next large CSR allocation.
        del functions, plans, plan, adjacency, cells, state, tasks, expected, output
        torch.cuda.empty_cache()
report['complete'] = True
bench.save(args.output, report)
