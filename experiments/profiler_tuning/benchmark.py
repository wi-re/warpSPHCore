#!/usr/bin/env python3
"""Screen profiler-guided candidates, then qualify fixed choices on held-out sizes.

All numerical checks and host reads happen before timing. Private measurements
include field packing but reuse scratch/output; public measurements allocate the
same scratch/output each call. Search, sorting and index conversion are excluded.
"""
import argparse
import ctypes as C
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--native-artifact', type=Path)
parser.add_argument('--precision', choices=['float32', 'float64'], required=True)
parser.add_argument('--dims', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--ns', type=int, nargs='+', default=[1000000])
parser.add_argument('--n-h', type=float, nargs='+', default=[4., 2.57])
parser.add_argument('--screen', action='store_true')
parser.add_argument('--selection', type=Path)
parser.add_argument('--repeat', type=int, default=7)
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--profile', help='Profile a selected name; no timing loop')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp', warpSPHCore_PRECISION=args.precision,
                  warpSPHCore_INDEX_DTYPE='int64', warpSPHCore_OFFSET_DTYPE='int64')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/benchmarks'), str(ROOT/'experiments/cluster_pairs')]
import torch
import warp as wp
import warpSPHCore as sph
import warpSPHCore.native._build as build
from warpSPHCore.native.api import radiusSearch, sphOperation
from runtime import Library, ClusterCase, build_cells, build_clusters, call as cluster_call
import bench_backends as bench

from api import CsrCase

torch.set_num_threads(16)
torch.set_num_interop_threads(1)
wp.init()
wp.set_module_options({'enable_backward': False}, module='warpSPHCore.coreOperations.native.wp_interpolate')
native_artifact = args.native_artifact or args.archive/'artifacts/optimized-native-torch.so'
torch.ops.load_library(str(native_artifact))
build._extension = torch.ops.warpsph_native
library = Library(args.library)
library.library.tuned_csr.argtypes = [C.POINTER(CsrCase)] + [C.c_int]*5
library.library.tuned_csr.restype = C.c_int
library.library.tuned_cluster.argtypes = [C.POINTER(ClusterCase)] + [C.c_int]*3
library.library.tuned_cluster.restype = C.c_int
selection = json.loads(args.selection.read_text())['selection'] if args.selection else {}
prior_selection = json.loads((args.archive/f'screen-{args.precision}.json').read_text())['selection']
properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
    operation=sph.WarpOperation.Interpolate, operationMode=sph.OperationDirection.TrueAllToToAll)
report = dict(complete=False, pid=os.getpid(), precision=args.precision, cases=[], activity=[], selection={},
    library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
    native_artifact=str(native_artifact),native_sha256=hashlib.sha256(native_artifact.read_bytes()).hexdigest(),
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.is_file()},
    environment=dict(torch=torch.__version__, warp=wp.__version__, gpu=torch.cuda.get_device_name()),
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})

def sample(function, iterations):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    begin.record()
    function(iterations)
    end.record()
    end.synchronize()
    return dict(event_ms=begin.elapsed_time(end)/iterations, wall_ms=(time.perf_counter()-start)*1000/iterations)

def many(function, iterations):
    for _ in range(iterations): result = function()
    return result

for dimension, nh, n in itertools.product(args.dims,args.n_h,args.ns):
    report['activity'].append(bench.activity())
    generator = torch.Generator().manual_seed(9067+n*3+dimension)
    x = torch.rand((n,dimension), generator=generator, dtype=torch.float64)
    side = round(n**(1/dimension))
    spacing = 1/side if side**dimension==n else n**(-1/dimension)
    x = x.to(device='cuda',dtype=getattr(torch,args.precision))
    h = x.new_full((n,),nh*spacing)
    state = sph.ParticleState(x,h,h.new_full((n,),1/n),torch.zeros(n,device='cuda',dtype=torch.int32),h.new_ones(n))
    values = torch.sin(x*2*math.pi)+.1*x
    domain = sph.DomainDescription(x.new_zeros(dimension),x.new_ones(dimension),torch.ones(dimension,device='cuda',dtype=torch.bool),dimension)
    cells = build_cells(library,state,values,float(h[0]),8)
    adjacency64 = radiusSearch(cells.state,domain)
    adjacency32 = sph.AdjacencyList(None,adjacency64.paddedJ.int(),adjacency64.numNeighbors,adjacency64.edgeOffsets,n,n,numEdges=adjacency64.numEdges)
    adjacencies = {64:adjacency64,32:adjacency32}
    native = lambda adj=adjacency64:sphOperation(cells.state,properties,domain,cells.values,adjacency=adj)
    native32 = lambda:sphOperation(cells.state,properties,domain,cells.values,adjacency=adjacency32)
    expected = native()
    tolerance = 3e-4 if args.precision=='float32' else 3e-9
    key = f'{args.precision}:{dimension}:{nh:g}'
    tile = prior_selection[key]['cluster_raw_super_shared'][:4]
    plan = build_clusters(library,cells,*tile,compression=False)
    packed_csr = x.new_empty((n,((2*dimension+3)//4)*4))
    packed_cluster = x.new_empty((2*dimension,n))
    output = torch.empty_like(cells.values)

    def csr_descriptor(bits, packed=packed_csr, out=output):
        request = CsrCase(dimension,int(args.precision=='float64'),n,bits)
        request.stream = library.stream
        tensors = {name:getattr(cells.state,name) for name in ['positions','supports','masses','densities']}
        adj = adjacencies[bits]
        tensors.update(values=cells.values,offsets=adj.edgeOffsets,indices=adj.paddedJ,counts=adj.numNeighbors,
                       lower=domain.min,upper=domain.max,periodic=domain.periodic,packed=packed,output=out)
        for name,tensor in tensors.items(): setattr(request,name,tensor.data_ptr())
        return request

    # Domain field names are explicit in the core dataclass; no host tensor reads.
    descriptors = {bits:csr_descriptor(bits) for bits in [32,64]}
    cluster_descriptor = plan.request(packed_cluster,output)
    owners = [packed_csr,packed_cluster,output,cluster_descriptor,*descriptors.values()]
    functions = {
        'native64_public':lambda count:many(native,count),
        'native32_public':lambda count:many(native32,count),
        'cluster_baseline':lambda count:library.run(cluster_descriptor,3,False,8,True,count)}
    if not args.screen:
        functions['cluster_baseline_public'] = lambda count:many(lambda:cluster_call(library,plan,3,False,8),count)
    configurations = {}
    if args.screen:
        candidates = list(itertools.product([1,2,4,8],[4,8,16,32],[128,256],[64,32]))
        variants = list(range(6))
    else:
        candidates = [selection[key]['csr64'],selection[key]['csr32']]
        variants = sorted(set([selection[key]['cluster']] + [int(name.split('_')[1]) for name in (args.profile or '').split(',') if name in [f'cluster_{v}' for v in range(6)]]))
    for batch,width,block,bits in candidates:
        name = f'csr_{batch}_{width}_{block}_{bits}'
        descriptor = descriptors[bits]
        def function(count,a=descriptor,b=batch,w=width,t=block):
            library.check(library.library.tuned_csr(C.byref(a),b,w,t,1,count))
            return output
        functions[name] = function
        configurations[name] = [batch,width,block,bits]
        if not args.screen:
            def public_csr(b=batch,w=width,t=block,bits=bits):
                packed = torch.empty_like(packed_csr)
                out = torch.empty_like(output)
                descriptor = csr_descriptor(bits,packed,out)
                library.check(library.library.tuned_csr(C.byref(descriptor),b,w,t,1,1))
                return out
            functions[name+'_public'] = lambda count,fn=public_csr:many(fn,count)
            configurations[name+'_public'] = [batch,width,block,bits]
    for variant in variants:
        name = f'cluster_{variant}'
        def function(count,v=variant):
            library.check(library.library.tuned_cluster(C.byref(cluster_descriptor),v,1,count))
            return output
        functions[name] = function
        configurations[name] = variant
        if not args.screen:
            def public_cluster(v=variant):
                packed = torch.empty_like(packed_cluster)
                out = torch.empty_like(output)
                descriptor = plan.request(packed,out)
                library.check(library.library.tuned_cluster(C.byref(descriptor),v,1,1))
                return out
            functions[name+'_public'] = lambda count,fn=public_cluster:many(fn,count)
            configurations[name+'_public'] = variant
    max_abs = {}
    for name,function in functions.items():
        result = function(1)
        if result is None: result=output
        torch.testing.assert_close(result,expected,rtol=tolerance,atol=tolerance)
        max_abs[name] = float((result-expected).abs().max())
        function(2)
    torch.cuda.synchronize()
    if args.profile:
        for name in args.profile.split(','):
            torch.cuda.nvtx.range_push(name)
            torch.cuda.cudart().cudaProfilerStart()
            functions[name](1)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.nvtx.range_pop()
        samples={}
    else:
        samples = {name:[] for name in functions}
        for repeat in range(args.repeat):
            names=list(functions)
            random.Random(6011+repeat).shuffle(names)
            for name in names: samples[name].append(sample(functions[name],args.iterations))
    records = [dict(name=name,configuration=configurations.get(name),samples=data,
                    event_ms=statistics.median(p['event_ms'] for p in data),
                    wall_ms=statistics.median(p['wall_ms'] for p in data),max_abs=max_abs[name])
               for name,data in samples.items()]
    if args.screen:
        choices={}
        for bits in [64,32]:
            best=min((r for r in records if r['name'].startswith('csr_') and r['configuration'][-1]==bits),key=lambda r:r['event_ms'])
            choices[f'csr{bits}']=best['configuration']
        choices['cluster']=min((r for r in records if r['name'].startswith('cluster_') and r['name']!='cluster_baseline'),key=lambda r:r['event_ms'])['configuration']
        report['selection'][key]=choices
    else: report['selection'][key]=selection[key]
    report['cases'].append(dict(dimension=dimension,n_h=nh,n=n,edges=adjacency64.numEdges,tile=tile,records=records))
    report['activity'].append(bench.activity())
    bench.save(args.output,report)
    print(key,n,report['selection'][key],flush=True)
    for r in sorted(records,key=lambda r:r['event_ms'])[:4]: print(r['name'],r['event_ms'],flush=True)
    del functions,descriptors,owners,plan,cells,adjacencies,adjacency64,adjacency32,expected,output
    torch.cuda.empty_cache()
report['complete']=True
bench.save(args.output,report)
