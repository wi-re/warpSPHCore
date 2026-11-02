"""Paired public-API timings in separate workers (Torch registrations are global).

Each worker keeps one immutable library loaded. The controller alternates them
serially; neither performs GPU work while the other is being measured. Sorting,
search, allocation of fixtures and output comparisons occur outside the samples.
"""
import argparse
from dataclasses import replace
import json
import importlib.util
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


OPERATIONS = ('density', 'interpolate', 'interpolate_vector', 'interpolate_matrix',
              'gradient', 'gradient_vector', 'gradient_matrix')


def emit(value):
    print(json.dumps(value), flush=True)


def worker(args):
    import torch
    from template_runtime import Plan
    torch.set_num_threads(24)
    torch.set_num_interop_threads(1)
    os.environ.update(WARPSPHCORE_BACKEND='native', warpSPHCore_PRECISION=args.precision,
        warpSPHCore_INDEX_DTYPE='int64', warpSPHCore_OFFSET_DTYPE='int64')
    sys.path.insert(0, str(args.core/'src'))
    import warpSPHCore as sph
    import warpSPHCore.native._build as build
    torch.ops.load_library(str(args.library))
    build._extension = torch.ops.warpsph_native
    assert sph.backend == 'native'
    spec = importlib.util.spec_from_file_location('production_morton_keys',
        args.core/'experiments/cell_tiles/runtime.py')
    morton = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = morton
    spec.loader.exec_module(morton)
    planner = morton.Library(args.planner)
    emit(dict(ready=True, pid=os.getpid(), torch=torch.__version__, backend=sph.backend))
    states = {}
    for line in sys.stdin:
        request = json.loads(line)
        if request['command'] == 'stop':
            return
        if request['command'] == 'prepare':
            # Release the preceding fixture before creating another large CSR.
            states.clear()
            started = time.perf_counter()
            n, dimension = request['n'], request['dimension']
            dtype = getattr(torch, args.precision)
            generator = torch.Generator().manual_seed(271828+n+dimension)
            positions = torch.rand((n,dimension),generator=generator,dtype=torch.float64).to('cuda',dtype=dtype)
            supports = positions.new_full((n,), request['nh']*n**(-1/dimension))
            # Sorting uses the old refined Morton convention, but needs no
            # shared-cell stencil. This also permits small periodic grids.
            grid = max(1, math.floor(1/float(supports[0])))
            refinement = 8
            while grid*refinement >= 2**21:
                refinement //= 2
            if not refinement:
                raise ValueError('Morton fixture resolution exceeds 21 bits')
            order = torch.argsort(planner.keys(positions,grid*refinement),stable=True)
            state = sph.ParticleState(positions[order],supports,supports.new_full((n,),1/n),
                torch.zeros(n,device='cuda',dtype=torch.int32),supports.new_ones(n))
            vector = torch.sin(state.positions*(2*math.pi))+.2*torch.cos(state.positions*(4*math.pi))
            scalar = (vector*torch.arange(1,dimension+1,device='cuda',dtype=dtype)).sum(-1,keepdim=True)
            domain = sph.DomainDescription(positions.new_zeros(dimension),positions.new_ones(dimension),
                torch.ones(dimension,device='cuda',dtype=torch.bool),dimension)
            adjacency = sph.radiusSearch(state,domain)
            csr = Plan(dict(indices=adjacency.paddedJ.to(torch.int64 if request['wide'] else torch.int32),
                counts=adjacency.numNeighbors,offsets=adjacency.edgeOffsets))
            planning = dict(edges=adjacency.numEdges,mean_neighbors=adjacency.numEdges/n,
                csr_bytes=csr.bytes(),morton_grid=grid,morton_refinement=refinement)
            del adjacency, positions, supports, order
            for device in ('cpu', 'cuda'):
                particles = replace(state, **{name: getattr(state, name).to(device)
                    for name in ('positions', 'supports', 'masses', 'densities', 'kinds')})
                local = csr.to(device)
                graph = sph.AdjacencyList(None, local.buffers['indices'], local.buffers['counts'],
                    local.buffers['offsets'], request['n'], request['n'], numEdges=planning['edges'])
                bounds = sph.DomainDescription(domain.min.to(device), domain.max.to(device),
                    domain.periodic.to(device), request['dimension'])
                v = vector.to(device)
                matrix = (v[:, :, None] * v[:, None, :]).contiguous()
                states[device] = (particles, bounds, graph, scalar.to(device)[:, 0], v, matrix)
            del state, scalar, vector, domain, csr
            torch.cuda.synchronize()
            planning['wall_ms'] = (time.perf_counter()-started)*1000
            emit(dict(planning=planning))
            continue

        particles, domain, graph, scalar, vector, matrix = states[request['device']]
        name = request['operation']
        operation = (sph.WarpOperation.Density if name == 'density' else
                     sph.WarpOperation.Gradient if name.startswith('gradient') else
                     sph.WarpOperation.Interpolate)
        values = matrix if name.endswith('matrix') else vector if name.endswith('vector') else scalar
        properties = sph.OperationProperties(sph.KernelFunctions.Wendland2,
            operation=operation, supportMode=sph.SupportScheme.Gather,
            gradientMode=sph.GradientScheme.Difference,
            operationMode=sph.OperationDirection.TrueAllToToAll)

        def call():
            return sph.warpOperation(particles, properties, domain,
                None if name == 'density' else values, adjacency=graph)

        if request['command'] == 'check':
            actual = call()
            rows = torch.linspace(0, len(actual)-1, min(257, len(actual)),
                device=actual.device).long()
            emit(dict(shape=list(actual.shape), sample=actual[rows].double().cpu().tolist()))
            continue
        gpu = request['device'] == 'cuda'
        affinity = os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else None
        # CPU submission time is visible between short GPU launches. On a
        # hybrid CPU, migrating one worker to an efficiency core can otherwise
        # look like a kernel regression. Use the same submitting core for both.
        if gpu and affinity is not None:
            os.sched_setaffinity(0,{min(affinity)})
        if request['command'] == 'prime':
            for _ in range(3):
                call()
        iterations = request.get('iterations', 3)
        if gpu:
            torch.cuda.synchronize()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
        started = time.perf_counter()
        for _ in range(iterations):
            call()
        if gpu:
            end.record()
            end.synchronize()
        result = dict(wall_ms=(time.perf_counter()-started)*1000/iterations)
        if gpu:
            result['event_ms'] = begin.elapsed_time(end)/iterations
        if gpu and affinity is not None:
            os.sched_setaffinity(0,affinity)
        emit(result)


class Worker:
    def __init__(self, args, library, precision, label):
        command = [sys.executable, __file__, '--worker', '--core', str(args.core),
            '--planner', str(args.planner), '--library', str(library), '--precision', precision]
        self.log = (args.output.parent / (args.output.stem+'-'+precision+'-'+label+'.log')).open('w')
        # An idle worker must sleep rather than spin on the other worker's
        # cores. Both libraries get the same OpenMP wait policy.
        environment = dict(os.environ, OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', KMP_BLOCKTIME='0')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1, env=environment)
        self.metadata = self.receive()

    def receive(self):
        while line := self.process.stdout.readline():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                self.log.write(line)
                self.log.flush()
        code = self.process.wait(timeout=10)
        raise RuntimeError(f'Worker {self.process.pid} exited {code}; see {self.log.name}')

    def ask(self, command, **arguments):
        self.process.stdin.write(json.dumps(dict(command=command, **arguments))+'\n')
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"stop"}\n')
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()


def activity(allowed):
    lines = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], text=True).strip().splitlines()
    others = [line for line in lines if int(line.split(',')[0]) not in allowed]
    if others:
        raise RuntimeError(f'Concurrent GPU workload: {others}')
    return dict(processes=lines, other_processes=others,
        gpu=subprocess.check_output(['nvidia-smi',
            '--query-gpu=utilization.gpu,clocks.sm,power.draw', '--format=csv,noheader'], text=True).strip())


def study(args):
    import torch
    torch.set_num_threads(1)  # Parent does only small result comparisons.
    report = dict(complete=False, fixtures=[], activity=[],
        libraries=dict(baseline=str(args.baseline), candidate=str(args.candidate)),
        cpu_threads=24, repeats=args.repeats, omp_wait_policy='PASSIVE', gomp_spincount=0,
        gpu_submit_cpu=min(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    rng = random.Random(20261102)
    for precision in args.precisions:
        workers = {}
        try:
            for label, library in [('baseline', args.baseline), ('candidate', args.candidate)]:
                workers[label] = Worker(args, library, precision, label)
            pids = {item.process.pid for item in workers.values()}
            for dimension in args.dimensions:
                for n in args.sizes:
                    for nh in args.neighborhoods:
                        for wide in ([False, True] if args.both_indices else [False]):
                            report['activity'].append(activity(pids))
                            config = dict(n=n, dimension=dimension, nh=nh, wide=wide)
                            current = dict(config, precision=precision, records=[], qualification=[])
                            report['fixtures'].append(current)
                            for label, process in workers.items():
                                current[label+'_planning'] = process.ask('prepare', **config)['planning']
                            print(f'{precision} D={dimension} N={n} nh={nh} wide={wide}', flush=True)
                            for device in ('cuda', 'cpu'):
                                for operation in args.operations:
                                    context = dict(device=device, operation=operation)
                                    checks = {label: process.ask('check', **context) for label, process in workers.items()}
                                    first, second = (torch.tensor(checks[label]['sample'], dtype=torch.float64)
                                        for label in ('baseline', 'candidate'))
                                    error = float((first-second).norm()/first.norm().clamp_min(1e-30))
                                    assert checks['baseline']['shape'] == checks['candidate']['shape']
                                    assert torch.isfinite(second).all() and error < (6e-5 if precision=='float32' else 2e-11), (context,error)
                                    current['qualification'].append(dict(context, error=error))
                                    iterations = {}
                                    for label, process in workers.items():
                                        measurement = process.ask('prime', **context)
                                        metric = measurement.get('event_ms', measurement['wall_ms'])
                                        iterations[label] = max(1, min(1000, int((75 if device=='cpu' else 15)/max(metric,.005))))
                                    samples = {label: [] for label in workers}
                                    for _ in range(args.repeats):
                                        labels = list(workers)
                                        rng.shuffle(labels)
                                        for label in labels:
                                            samples[label].append(workers[label].ask('measure', **context,
                                                iterations=iterations[label]))
                                    for label, rows in samples.items():
                                        current['records'].append(dict(context, path=label, iterations=iterations[label],
                                            samples=rows, **{key:statistics.median(row[key] for row in rows) for key in rows[0]}))
                                    save()
                            report['activity'].append(activity(pids))
                            current['complete'] = True
                            save()
        finally:
            for process in workers.values():
                process.close()
    report['complete'] = True
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--planner', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--library', type=Path)
    parser.add_argument('--precision', default='float32')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--precisions', nargs='+', default=['float32','float64'])
    parser.add_argument('--dimensions', nargs='+', type=int, default=[1,2,3])
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024,32768,262144,1048576])
    parser.add_argument('--neighborhoods', nargs='+', type=float, default=[4.,2.57])
    parser.add_argument('--operations', nargs='+', choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument('--both-indices', action='store_true')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    worker(args) if args.worker else study(args)
