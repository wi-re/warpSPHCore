#!/usr/bin/env python3
"""Screen lossless encodings, then measure fixed schedules on held-out sizes.

All search/sorting/compression, validation, conversions and readbacks happen
before interpolation timings. Matched raw controls use the same private launch
path. Public native/Torch controls provide context for wrapper/allocator cost.
"""
import argparse
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--precision', choices=('float32','float64'), required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--screen', action='store_true')
parser.add_argument('--selection', type=Path)
parser.add_argument('--byte-selection',type=Path,required=True)
parser.add_argument('--ns', type=int, nargs='+', default=[1000000])
parser.add_argument('--dims', type=int, nargs='+', default=[1,2,3])
parser.add_argument('--n-h', type=float, nargs='+', default=[4.,2.57])
parser.add_argument('--repeat', type=int, default=9)
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--target-ms', type=float, help='Calibrate 1..100 calls per timing batch from a native64 pilot')
parser.add_argument('--plan-repeat', type=int, default=3)
parser.add_argument('--profile', help='Comma-separated selected private paths; skips timing')
args = parser.parse_args()
os.environ.update(WARPSPHCORE_BACKEND='warp', warpSPHCore_PRECISION=args.precision,
                  warpSPHCore_INDEX_DTYPE='int64', warpSPHCore_OFFSET_DTYPE='int64')
sys.path.insert(0, str(ROOT/'src'))
import torch
import warpSPHCore as sph
import warpSPHCore.native._build as build
from warpSPHCore.native.api import radiusSearch, sphOperation
from gaps_runtime import Library, RAW64, RAW32, DELTA, CELL, BAND, NIBBLE, NAMES, build_plan, descriptor
from runtime import Library as ByteLibrary, build_plan as build_byte_plan
NAMES = dict(NAMES, **{})
NAMES.update({DELTA:'delta',CELL:'cell'})

spec = importlib.util.spec_from_file_location('compression_planner', ROOT/'experiments/cluster_pairs/runtime.py')
planner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner
spec.loader.exec_module(planner)
library = Library(args.library)
cells_library = planner.Library(args.artifacts/'exploration-v2.so')
native_artifact = args.artifacts/'native-torch-final.so'
torch.ops.load_library(str(native_artifact))
build._extension = torch.ops.warpsph_native
torch.set_num_threads(16)
torch.set_num_interop_threads(1)
selection = json.loads(args.selection.read_text())['selection'] if args.selection else {}
byte_selection = json.loads(args.byte_selection.read_text())['selection']
properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
    operation=sph.WarpOperation.Interpolate, operationMode=sph.OperationDirection.TrueAllToToAll)


def activity():
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True).strip()
    other = [line for line in raw.splitlines() if line.split(',')[0].strip() != str(os.getpid())]
    clocks = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,clocks.mem,power.draw',
                                      '--format=csv,noheader'], text=True).strip()
    return dict(processes=raw, other_processes=other, gpu=clocks)


def measure(function, iterations):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    begin.record()
    function(iterations)
    end.record()
    end.synchronize()
    return dict(event_ms=begin.elapsed_time(end)/iterations, wall_ms=(time.perf_counter()-start)*1000/iterations)


def many(function, count):
    for _ in range(count):
        result = function()
    return result


report = dict(complete=False, precision=args.precision, pid=os.getpid(), cases=[], selection={}, activity=[],
    configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
    environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name()),
    sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [args.library,native_artifact,args.artifacts/'exploration-v2.so',*Path(__file__).parent.glob('*.py'),
             Path(__file__).parent/'kernels.cu',ROOT/'experiments/csr_compression/kernels.cu']})
for dimension, nh, n in itertools.product(args.dims,args.n_h,args.ns):
    report['activity'].append(activity())
    generator = torch.Generator().manual_seed(9067+n*3+dimension)
    x = torch.rand((n,dimension), generator=generator, dtype=torch.float64)
    side = round(n**(1/dimension))
    spacing = 1/side if side**dimension == n else n**(-1/dimension)
    x = x.to(device='cuda',dtype=getattr(torch,args.precision))
    h = x.new_full((n,),nh*spacing)
    state = sph.ParticleState(x,h,h.new_full((n,),1/n),torch.zeros(n,device='cuda',dtype=torch.int32),h.new_ones(n))
    values = torch.sin(x*2*math.pi)+.1*x
    domain = sph.DomainDescription(x.new_zeros(dimension),x.new_ones(dimension),
                                  torch.ones(dimension,device='cuda',dtype=torch.bool),dimension)
    print(json.dumps(dict(phase='sorting',dimension=dimension,nh=nh,n=n)),flush=True)
    if dimension == 1:
        # Coordinate order is exact Morton order in 1D. Bypass the toy key
        # encoder's 21-bit limit and retain the physical support-sized cells.
        order = torch.argsort(state.positions[:,0],stable=True)
        ordered = replace(state,**{name:getattr(state,name)[order] for name in
                                   ('positions','supports','masses','densities','kinds')})
        grid = math.floor(1/float(h[0]))
        cell_keys = (ordered.positions[:,0]*grid).clamp(max=grid-1).long()
        _,cell_counts = torch.unique_consecutive(cell_keys,return_counts=True)
        cell_offsets = torch.cat((cell_counts.new_zeros(1),cell_counts.cumsum(0)))
        cells = SimpleNamespace(state=ordered,values=values[order],order=order,counts=cell_counts,
                                offsets=cell_offsets,grid=grid,refinement='exact coordinate order')
    else:
        cells = planner.build_cells(cells_library,state,values,float(h[0]),8)
    print(json.dumps(dict(phase='search',dimension=dimension,nh=nh,n=n)),flush=True)
    adjacency = radiusSearch(cells.state,domain)
    print(json.dumps(dict(phase='encoding',edges=adjacency.numEdges,torch_allocated_bytes=torch.cuda.memory_allocated())),flush=True)
    indices32 = adjacency.paddedJ.int()
    adjacency32 = sph.AdjacencyList(None,indices32,adjacency.numNeighbors,adjacency.edgeOffsets,n,n,numEdges=adjacency.numEdges)
    native64 = lambda:sphOperation(cells.state,properties,domain,cells.values,adjacency=adjacency)
    native32 = lambda:sphOperation(cells.state,properties,domain,cells.values,adjacency=adjacency32)
    expected = native64()
    packed = x.new_empty((n,((2*dimension+3)//4)*4))
    output = torch.empty_like(cells.values)
    tensors = {name:getattr(cells.state,name) for name in ('positions','supports','masses','densities')}
    tensors.update(values=cells.values,lower=domain.min,upper=domain.max,periodic=domain.periodic,
                   packed=packed,output=output)
    key = f'{args.precision}:{dimension}:{nh:g}'
    configurations = {encoding:(list(itertools.product((4,8,16,32),(1,2,4),(1,2),(128,256)))
                                if args.screen else [selection[key][NAMES[encoding]]])
                      for encoding in (RAW64,RAW32,BAND,NIBBLE)}
    configurations.update({encoding:[byte_selection[key][NAMES[encoding]]] for encoding in (DELTA,CELL)})
    plans, planning = {}, []
    for encoding in (DELTA,CELL,BAND,NIBBLE):
        for width in sorted({config[0] for config in configurations[encoding]}):
            plan_times = []
            for repeat in range(1 if args.screen else args.plan_repeat):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if encoding in (DELTA,CELL):
                    # The earlier byte encoder uses the base class's ABI.
                    byte_library = ByteLibrary(args.library)
                    plan = build_byte_plan(byte_library,adjacency.paddedJ,adjacency.numNeighbors,
                                           adjacency.edgeOffsets,width,encoding,cells.offsets,n)
                else:
                    plan = build_plan(library,adjacency.paddedJ,adjacency.numNeighbors,
                                      adjacency.edgeOffsets,width,encoding)
                torch.cuda.synchronize()
                plan_times.append((time.perf_counter()-start)*1000)
            plans[encoding,width] = plan
            planning.append(dict(format=NAMES[encoding],width=width,wall_ms=plan_times,storage=plan.storage()))
    functions, config_by_name, owners = {}, {}, []
    for encoding, configs in configurations.items():
        for config in configs:
            width,items,batch,threads = config if encoding not in (DELTA,CELL) else (config[0],4 if config[3] else 1,config[1],config[2])
            vector = bool(config[3]) if encoding in (DELTA,CELL) else False
            name = f'{NAMES[encoding]}_{width}_{items}_{batch}_{threads}' if args.screen else NAMES[encoding]
            def request_for(packed=packed,out=output,encoding=encoding,width=width):
                common = dict(tensors,packed=packed,output=out)
                if encoding >= DELTA:
                    return plans[encoding,width].request(common,n,dimension,int(args.precision=='float64'))
                return descriptor(dict(common,counts=adjacency.numNeighbors,row_offsets=adjacency.edgeOffsets,
                                       indices=adjacency.paddedJ if encoding == RAW64 else indices32),n,
                                  dimension=dimension,precision=int(args.precision=='float64'))
            request = request_for()
            owners.append(request)
            def function(count,request=request,encoding=encoding,width=width,batch=batch,threads=threads,vector=vector,items=items):
                if encoding in (DELTA,CELL):
                    ByteLibrary.run(library,request,encoding,width,batch,threads,True,count,vector)
                else:
                    library.run(request,encoding,width,items,batch,threads,True,count)
                return output
            functions[name] = function
            config_by_name[name] = list(config)
            if not args.screen and encoding >= DELTA:
                def public(request_for=request_for,encoding=encoding,width=width,batch=batch,threads=threads,vector=vector,items=items):
                    local_packed = torch.empty_like(packed)
                    local_output = torch.empty_like(output)
                    request = request_for(local_packed,local_output)
                    if encoding in (DELTA,CELL):
                        ByteLibrary.run(library,request,encoding,width,batch,threads,vector=vector)
                    else:
                        library.run(request,encoding,width,items,batch,threads)
                    return local_output
                functions[name+'_public'] = lambda count,fn=public:many(fn,count)
                config_by_name[name+'_public'] = list(config)
    functions.update(native64_public=lambda count:many(native64,count),
                     native32_public=lambda count:many(native32,count))
    max_abs = {}
    tolerance = 3e-4 if args.precision == 'float32' else 3e-9
    for name,function in functions.items():
        result = function(1)
        torch.testing.assert_close(result,expected,rtol=tolerance,atol=tolerance)
        max_abs[name] = float((result-expected).abs().max())
        function(2)
    iterations = args.iterations
    calibration = None
    if args.target_ms:
        calibration = measure(functions['native64_public'],min(3,args.iterations))
        iterations = min(100,max(1,round(args.target_ms/calibration['event_ms'])))
    print(json.dumps(dict(phase='timing',dimension=dimension,nh=nh,n=n,iterations=iterations)),flush=True)
    if args.profile:
        for name in args.profile.split(','):
            torch.cuda.nvtx.range_push(name)
            torch.cuda.cudart().cudaProfilerStart()
            functions[name](1)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.nvtx.range_pop()
        samples = {}
    else:
        samples = {name:[] for name in functions}
        for repeat in range(args.repeat):
            names = list(functions)
            random.Random(51031+repeat).shuffle(names)
            for name in names:
                samples[name].append(measure(functions[name],iterations))
    records = [dict(name=name,configuration=config_by_name.get(name),samples=data,
                    event_ms=statistics.median(p['event_ms'] for p in data),
                    wall_ms=statistics.median(p['wall_ms'] for p in data),max_abs=max_abs[name])
               for name,data in samples.items()]
    if args.screen:
        report['selection'][key] = {NAMES[encoding]:min(
            (record for record in records if record['name'].startswith(NAMES[encoding]+'_')),
            key=lambda record:record['event_ms'])['configuration'] for encoding in (RAW64,RAW32,BAND,NIBBLE)}
    report['cases'].append(dict(dimension=dimension,nh=nh,particles=n,edges=adjacency.numEdges,
        mean_neighbors=adjacency.numEdges/n,planning=planning,records=records,
        iterations=iterations,calibration=calibration,sorting_grid=cells.grid,sorting_refinement=cells.refinement,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        csr32_bytes=indices32.numel()*4+adjacency.numNeighbors.numel()*4+adjacency.edgeOffsets.numel()*8,
        csr64_bytes=adjacency.paddedJ.numel()*8+adjacency.numNeighbors.numel()*4+adjacency.edgeOffsets.numel()*8))
    report['activity'].append(activity())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    summary = report['selection'].get(key) if args.screen else {r['name']:round(r['event_ms'],6) for r in records}
    print(json.dumps(dict(key=key,n=n,selection_or_event_ms=summary)),flush=True)
    del functions, plans, planning, owners, request, cells, adjacency, adjacency32, indices32, expected, packed, output, tensors, state, values,x,h
    torch.cuda.empty_cache()
report['complete'] = True
report['uncontended_process_snapshots'] = all(not sample['other_processes'] for sample in report['activity'])
args.output.write_text(json.dumps(report,indent=2)+'\n')
