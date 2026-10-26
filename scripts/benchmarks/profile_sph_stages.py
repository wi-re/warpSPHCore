#!/usr/bin/env python3
"""Separate CUDA kernel trace and live resource census for core SPH stages.

Instrumented durations are diagnostic, not public-call performance samples.
Generic Warp overloads are observed from actual launches; driver resource
queries run after warmup and outside the profiler. Torch/vendor kernels remain
in the trace, including those hidden behind CRK and renormalization wrappers.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from _operator_cases import OPERATIONS, Case, Workload, check_output
from bench_core_operators import qualify, save, source_hashes


@contextmanager
def observe_launches():
    import warp as wp
    observed = {}
    original = wp.launch
    def launch(kernel, dim, inputs=None, outputs=None, **kw):
        args = list(inputs or []) + list(outputs or [])
        concrete = kernel.add_overload(kernel.infer_argument_types(args)) if kernel.is_generic else kernel
        block = kw.get('block_dim', concrete.module.options.get('block_dim', 256))
        observed[(concrete.module.name, concrete.key, block)] = concrete
        return original(kernel, dim, inputs=inputs or [], outputs=outputs or [], **kw)
    wp.launch = launch
    try:
        yield observed
    finally:
        wp.launch = original


def summarize(events, repeat):
    kernels, runtime = defaultdict(list), Counter()
    cpu = Counter()
    cpu_durations = defaultdict(list)
    for event in events:
        category = event.get('cat')
        if category == 'kernel':
            kernels[event['name']].append(event)
        elif category in ('cuda_runtime', 'cuda_driver'):
            runtime[event['name']] += 1
        elif category == 'cpu_op':
            cpu[event['name']] += 1
            cpu_durations[event['name']].append(event.get('dur', 0))
    return dict(kernel_launches_per_call=sum(map(len, kernels.values())) / repeat,
                summed_kernel_ms_per_call=sum(e['dur'] for es in kernels.values() for e in es) / (1000 * repeat),
                cuda_runtime_counts=dict(runtime), cpu_op_counts=dict(cpu),
                cpu_inclusive_ms_per_call={name: sum(values) / (1000 * repeat) for name, values in cpu_durations.items()},
                cpu_duration_scope='inclusive per named CPU operation; nested operations overlap and must not be summed',
                kernels=[dict(name=name, count=len(es), total_ms=sum(e['dur'] for e in es) / 1000,
                              mean_ms_per_call=sum(e['dur'] for e in es) / (1000 * repeat),
                              launch_attributes=es[0].get('args', {}))
                         for name, es in sorted(kernels.items(), key=lambda pair: -sum(e['dur'] for e in pair[1]))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ns', nargs='+', type=int, default=[1024, 65536])
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--neighbors', type=int, default=55)
    p.add_argument('--operations', nargs='+', choices=OPERATIONS,
                   default=['renorm', 'crk_factors', 'crk_terms', 'pinv_symmetric', 'torch_pinv'])
    p.add_argument('--traversals', nargs='+', choices=['adjacency', 'grid'], default=['adjacency'])
    p.add_argument('--lanes', nargs='+', type=int, choices=[1, 2, 4, 8, 16, 32], default=[1])
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.repeat <= 0:
        p.error('repeat must be positive')
    import torch
    import warp as wp
    from warpSPHCore import setNeighborLanes, get_torch_precision
    if not torch.cuda.is_available():
        p.error('requires CUDA')
    wp.config.kernel_cache_dir = str(a.output.parent / 'warp-cache')
    wp.init()
    torch.set_num_threads(2)
    hashes = source_hashes()
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(complete=False, scope=__doc__, command=sys.argv, records=[], source_sha256=hashes,
                  profiler_script_sha256=own_hash, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, warp=wp.__version__, dtype=str(get_torch_precision()), repeat=a.repeat,
                  synchronization_scope='runtime counts include one explicit end-of-profile synchronization; no output validation inside capture')
    save(a.output, report)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            for n in a.ns:
                w = Workload(n, a.dim, a.neighbors, 'cuda:0')
                for op in a.operations:
                    for traversal in a.traversals:
                        for lanes in a.lanes:
                            c = Case(n, a.dim, a.neighbors, op, traversal, lanes)
                            try:
                                call, diagnostics = qualify(w, c)
                            except AssertionError as error:
                                report['records'].append(dict(**c.dictionary(), status='failed_numerical_check',
                                                              error=f'{type(error).__name__}: {error}'))
                                save(a.output, report)
                                print(f'{op} N={n} L{lanes}: failed_numerical_check; no trace collected', flush=True)
                                continue
                            with observe_launches() as observed:
                                for _ in range(4):
                                    output = call()
                            stream.synchronize()
                            properties = []
                            for (module, key, block), kernel in observed.items():
                                row = dict(module=module, kernel=key, block_dim=block)
                                try:
                                    row.update(wp.get_cuda_kernel_properties(kernel, device='cuda:0', block_dim=block))
                                except Exception as error:
                                    row['resource_query_error'] = str(error)
                                properties.append(row)
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                     torch.profiler.ProfilerActivity.CUDA]) as prof:
                                for _ in range(a.repeat):
                                    output = call()
                                stream.synchronize()
                            check_output(output)
                            stem = f'd{a.dim}-n{n}-{op}-{traversal}-l{lanes}-{len(report["records"])}'
                            trace = a.output.parent / 'traces' / f'{a.output.stem}-{stem}.json'
                            trace.parent.mkdir(parents=True, exist_ok=True)
                            prof.export_chrome_trace(str(trace))
                            events = json.loads(trace.read_text())['traceEvents']
                            report['records'].append(dict(**c.dictionary(), status='profiled', qualification=diagnostics,
                                                          trace=str(trace.resolve()), live_warp_properties=properties,
                                                          **summarize(events, a.repeat)))
                            save(a.output, report)
                            print(f'{stem}: {report["records"][-1]["kernel_launches_per_call"]:g} CUDA kernels/call', flush=True)
        if hashes != source_hashes() or own_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError('source changed during profiling')
        report['complete'] = True
    finally:
        setNeighborLanes(None)
        save(a.output, report)


if __name__ == '__main__':
    main()
