#!/usr/bin/env python3
"""Qualify and time private 2D vector interpolation candidates on identical CSR.

CUDA events include preprocessing launches and completed wall time includes the
final stream wait. Scratch/output allocation is excluded from private timings;
production promotion requires public native/Torch timing with allocations.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--precision', choices=['float32', 'float64'], default='float32')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeat', type=int, default=7)
parser.add_argument('--iterations', type=int, default=20)
parser.add_argument('--native-artifact',type=Path)
parser.add_argument('--profile', help='One candidate as mode,lanes,block, or native/warp')
parser.add_argument('--modes', type=int, nargs='+', default=[0,1,2,3,4,5,6,8,9,10,11])
args = parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp', warpSPHCore_PRECISION=args.precision,
                  warpSPHCore_INDEX_DTYPE='int64', warpSPHCore_OFFSET_DTYPE='int64')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/benchmarks')]
import bench_backends as bench
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.native.api import sphOperation
bench.torch, bench.sph = torch, sph
if args.native_artifact:
    import warpSPHCore.native._build as build
    torch.ops.load_library(str(args.native_artifact.resolve()))
    build._extension=torch.ops.warpsph_native
torch.set_num_threads(16)
wp.init()
wp.set_module_options({'enable_backward':False},
                      module='warpSPHCore.coreOperations.native.wp_interpolate')
data = torch.load(args.fixture, weights_only=True)
n, edges = data['n'], data['edges']
dtype = getattr(torch, args.precision)
def tensor(name):
    value = data[name]
    return value.to(device='cuda', dtype=dtype if value.is_floating_point() else value.dtype)
q = sph.ParticleState(*(tensor(k) for k in ['positions','supports','masses','kinds','densities']))
domain = sph.DomainDescription(*(tensor(k) for k in ['lower','upper','periodic']), 2)
adjacency = sph.AdjacencyList(None, *(tensor(k) for k in ['indices','counts','offsets']),
                             n, n, numEdges=edges)
values = tensor('values')
properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
    operation=sph.WarpOperation.Interpolate, gradientMode=sph.GradientScheme.Difference,
    laplacianMode=sph.LaplacianScheme.Brookshaw,
    operationMode=sph.OperationDirection.TrueAllToToAll)
output = torch.empty_like(values)
temporary = torch.empty((n,4),device=values.device,dtype=values.dtype)
request = bench.native_request(q, domain, values, adjacency, output,
                               sph.WarpOperation.Interpolate, 16)
lib = C.CDLL(str(args.library.resolve()))
lib.value_variant_batch.argtypes = [C.POINTER(bench.NativeCase), C.c_int, C.c_int, C.c_int,
                                    C.c_void_p, C.c_int]
lib.value_variant_batch.restype = C.c_int

def launch(candidate, iterations=1):
    status = lib.value_variant_batch(C.byref(request), *candidate,
                                     temporary.data_ptr(), iterations)
    if status:
        raise RuntimeError(f'CUDA error {status} for {candidate}')

def native():
    return sphOperation(q, properties, domain, values, adjacency=adjacency)
def warp():
    return sph.warpOperation(q, properties, domain, values, adjacency=adjacency)

if args.profile:
    if args.profile == 'native': native()
    elif args.profile == 'warp': warp()
    else: launch(tuple(map(int,args.profile.split(','))))
    torch.cuda.synchronize()
    sys.exit(0)

expected = native()
reference = warp()
tolerance = 3e-5 if dtype == torch.float32 else 3e-9
torch.testing.assert_close(expected, reference, rtol=tolerance, atol=tolerance)
candidates = [(mode, width, block) for mode in args.modes
              for width in [1,2,4,8,16,32] for block in [128,256]]
candidates += [(0,8,64)] if 0 in args.modes else []
errors = {}
for candidate in candidates:
    launch(candidate)
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    errors[candidate] = (output-expected).abs().max().item()

report = dict(complete=False, checkpoint='0560de5', pid=os.getpid(), precision=args.precision, n=n, edges=edges,
    library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__),Path(__file__).with_name('variants.cu')]},
    configuration=vars(args)|dict(native_artifact=str(args.native_artifact) if args.native_artifact else None,library=str(args.library),fixture=str(args.fixture),output=str(args.output)),
    scope='Private kernel batches; resident input/scratch/output; prepare launches included; allocation excluded',
    activity=[bench.activity()], records=[])
samples = {c:[] for c in candidates}
args.output.parent.mkdir(parents=True,exist_ok=True)
for repeat in range(args.repeat):
    order = candidates.copy()
    random.Random(447+repeat).shuffle(order)
    for candidate in order:
        launch(candidate,3)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        launch(candidate,args.iterations)
        end.record()
        end.synchronize()
        wall = (time.perf_counter()-before)*1000/args.iterations
        samples[candidate].append(dict(kernel_ms=start.elapsed_time(end)/args.iterations, wall_ms=wall))
    report['activity'].append(bench.activity())
for candidate, points in samples.items():
    times = [p['kernel_ms'] for p in points]
    report['records'].append(dict(mode=candidate[0],lanes=candidate[1],block=candidate[2],
        median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),samples=points,
        max_abs=errors[candidate],temporary_bytes=n*(4 if candidate[0]==11 else (2 if candidate[0] in [4,9] else 1))*output.element_size()
        if candidate[0] in [3,4,9,10,11] else 0))
report['complete'] = True
args.output.write_text(json.dumps(report,indent=2)+'\n')
for record in sorted(report['records'],key=lambda r:r['median_ms'])[:16]:
    print({k:record[k] for k in ['mode','lanes','block','median_ms','min_ms','max_ms','temporary_bytes']})
print('Baseline:', next(r['median_ms'] for r in report['records']
                         if (r['mode'],r['lanes'],r['block'])==(0,8,128)))
