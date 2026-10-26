#!/usr/bin/env python3
"""Plan, qualify or time core SPH operators; no linear solves.

plan imports no Torch/Warp and touches no GPU. validate defaults to CPU.
benchmark is explicit CUDA-only, synchronized public-call wall timing with
separate CUDA event intervals (including launch gaps), never profiler timing.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from _operator_cases import CORE, LOCAL, OPERATIONS, Case, Workload, cases, check_output, work_counts


def source_hashes():
    paths = sorted((ROOT / 'src/warpSPHCore').rglob('*.py'))
    paths += sorted(Path(__file__).parent.glob('*operator*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['plan', 'validate', 'benchmark'])
    p.add_argument('--ns', nargs='+', type=int, default=[1024, 4096, 16384, 65536, 262144, 1048576])
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--neighbors', nargs='+', type=int, default=[25, 55, 110, 220])
    p.add_argument('--operations', nargs='+', choices=OPERATIONS, default=list(OPERATIONS))
    p.add_argument('--traversals', nargs='+', choices=['adjacency', 'grid', 'rebuild'], default=['adjacency', 'grid'])
    p.add_argument('--lanes', nargs='+', type=int, choices=[1, 2, 4, 8, 16, 32], default=[1, 32])
    p.add_argument('--forms', nargs='+', choices=['plain', 'renorm', 'crk'], default=['plain'])
    p.add_argument('--kernels', nargs='+', choices=['Wendland2', 'Wendland4', 'Wendland6', 'CubicSpline', 'QuinticSpline'], default=['Wendland2'])
    p.add_argument('--supports', nargs='+', choices=['Gather', 'Scatter', 'MeanSymmetric', 'KernelMeanSymmetric', 'SuperSymmetric', 'PartialSymmetric'], default=['Gather'])
    p.add_argument('--gradients', nargs='+', choices=['Naive', 'Symmetric', 'Difference', 'Summation'], default=['Difference'])
    p.add_argument('--laplacians', nargs='+', choices=['Naive', 'Brookshaw', 'Dot', 'Default'], default=['Brookshaw'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--repeat', type=int, default=9)
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--max-estimated-call-ms', type=float, help='skip larger cases projected above this time from a measured smaller case in the same series')
    p.add_argument('--keep-going', action='store_true', help='record numerical check failures and continue other cases; failed cases are excluded from plots')
    p.add_argument('--profile-range', action='store_true', help='enable CUDA profiler range around one qualified call per case; no timing samples')
    p.add_argument('--output', type=Path, required=True)
    return p


def measure(call, repeat, warmup):
    import torch
    for _ in range(warmup):
        output = call()
    torch.cuda.current_stream().synchronize()
    walls, events = [], []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        output = call()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - before) * 1000)
        events.append(start.elapsed_time(end))
    return output, dict(wall_ms=walls, median_wall_ms=statistics.median(walls), device_interval_ms=events)


def qualify(workload, c):
    import torch
    from warpSPHCore import setNeighborLanes
    refcall = workload.call(replace(c, traversal='adjacency', lanes=1))
    setNeighborLanes(1)
    reference = refcall()
    call = workload.call(c)  # may prepare frozen corrections, resetting lanes
    setNeighborLanes(c.lanes)
    output = call()
    # CRK moment count currently counts candidate visits on grid, exact CSR
    # entries on adjacency. Preserve this diagnostic mismatch in each record.
    check_output(output[:6], reference[:6]) if c.operation == 'crk_moments' else check_output(output, reference)
    if c.operation in ('pinv_symmetric', 'apply_pinv', 'torch_pinv'):
        truth = torch.linalg.pinv(workload.matrix.double(), rtol=1e-6).to(workload.matrix.dtype)
        if c.operation == 'apply_pinv':
            truth = workload.vector[:, None, :] @ truth
        check_output(output, truth)
    diagnostics = {}
    if c.operation == 'crk_moments':
        diagnostics = dict(returned_neighbor_count_sum=int(output[-1].sum()),
                           csr_neighbor_count_sum=int(reference[-1].sum()),
                           neighbor_count_matches_csr=bool(torch.equal(output[-1], reference[-1])),
                           count_scope='current CRK kernel candidate visits, not physical pair normalization')
    return call, diagnostics


def main():
    p = parser()
    a = p.parse_args()
    if a.max_estimated_call_ms is not None and a.max_estimated_call_ms <= 0:
        p.error('max-estimated-call-ms must be positive')
    if a.repeat < 2 or a.warmup < 1:
        p.error('repeat >= 2 and warmup >= 1 required')
    try:
        matrix = list(cases(a.ns, **{k: getattr(a, k) for k in ('dim', 'neighbors', 'operations', 'traversals', 'lanes', 'forms', 'kernels', 'supports', 'gradients', 'laplacians')}))
    except ValueError as e:
        p.error(str(e))
    report = dict(schema_version=1, mode=a.mode, complete=False, configuration=vars(a) | {'output': str(a.output)},
                  source_sha256=source_hashes(), records=[], command=sys.argv,
                  timing_scope='synchronized public call; fixed inputs; setup excluded except traversal=rebuild',
                  correction_scope='corrected consumers use frozen factors; full renorm/CRK calls include factor construction')
    if a.mode == 'plan':
        report.update(complete=True, case_count=len(matrix), cases=[c.dictionary() for c in matrix])
        save(a.output, report)
        print(f'{len(matrix)} cases -> {a.output}')
        return
    import torch
    import warp as wp
    from warpSPHCore import setNeighborLanes
    from warpSPHCore.type_config import get_torch_precision
    if a.mode == 'benchmark':
        if not a.device.startswith('cuda') or not torch.cuda.is_available():
            p.error('benchmark requires --device cuda and an accessible CUDA device')
        torch.cuda.set_device(a.device)
    elif a.device == 'cpu' and any(c.lanes != 1 for c in matrix):
        p.error('CPU qualification requires --lanes 1; tiled shuffle execution needs CUDA')
    wp.config.kernel_cache_dir = str(a.output.parent / 'warp-cache')
    wp.init()
    torch.set_num_threads(2)
    report.update(torch=torch.__version__, warp=wp.__version__, python=platform.python_version(), dtype=str(get_torch_precision()),
                  gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None)
    save(a.output, report)
    stream = torch.cuda.Stream() if a.device.startswith('cuda') else None
    from contextlib import nullcontext
    try:
        with torch.cuda.stream(stream) if stream else nullcontext():
            last_key, workload = None, None
            measured_costs = {}
            for c in matrix:
                key = (c.n, c.dim, c.neighbors)
                if key != last_key:
                    call = output = None
                    del workload
                    workload = Workload(*key, a.device)
                    last_key = key
                row = c.dictionary()
                row['field_shape'] = [c.dim] if c.operation in ('divergence', 'curl') or (c.operation == 'laplacian' and c.laplacian == 'Dot') else []
                series_key = tuple((k, str(v)) for k, v in c.dictionary().items() if k != 'n')
                prior = measured_costs.get(series_key)
                estimate = None if prior is None else prior[1] * c.n / prior[0]
                if a.max_estimated_call_ms is not None and estimate is not None and estimate > a.max_estimated_call_ms:
                    row.update(status='skipped_estimated_cost', estimated_call_ms=estimate,
                               estimate_basis=dict(n=prior[0], measured_wall_ms=prior[1]),
                               reason='linear work-size projection exceeds explicit per-call budget; no timing sample collected')
                    report['records'].append(row)
                    save(a.output, report)
                    print(f'{c.operation} N={c.n}: skipped projected {estimate:.1f} ms', flush=True)
                    continue
                try:
                    call, qualification = qualify(workload, c)
                    row['qualification'] = qualification
                    row.update(status='qualified', edges=workload.edges, degree=workload.edges / c.n,
                               counts=work_counts(c, workload.edges))
                    if a.mode == 'benchmark':
                        torch.cuda.reset_peak_memory_stats()
                        if a.profile_range:
                            for _ in range(a.warmup):
                                call()
                            stream.synchronize()
                            wp.cuda_profiler_start()
                            output = call()
                            stream.synchronize()
                            wp.cuda_profiler_stop()
                            check_output(output)
                            row['status'] = 'profiled_without_timing'
                        else:
                            output, timing = measure(call, a.repeat, a.warmup)
                            check_output(output)
                            row.update(timing)
                            measured_costs[series_key] = (c.n, timing['median_wall_ms'])
                            count = row['counts']['evaluated_nonself_pairs']
                            row['interactions_per_second'] = None if count is None else count * 1000 / timing['median_wall_ms']
                            row['matrices_per_second'] = None if row['counts']['matrices'] is None else c.n * 1000 / timing['median_wall_ms']
                        row['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
                        row['peak_torch_reserved_bytes'] = torch.cuda.max_memory_reserved()
                        row['free_device_bytes'], row['total_device_bytes'] = torch.cuda.mem_get_info()
                    report['records'].append(row)
                except AssertionError as e:
                    row.update(status='failed_numerical_check', error=f'{type(e).__name__}: {e}')
                    report['records'].append(row)
                    if not a.keep_going:
                        raise
                except Exception as e:
                    row.update(status='failed', error=f'{type(e).__name__}: {e}')
                    report['records'].append(row)
                    raise
                finally:
                    save(a.output, report)
                print(f'{c.operation}/{c.form}/{c.traversal}/L{c.lanes} N={c.n}: {row["status"]}', flush=True)
        if source_hashes() != report['source_sha256']:
            raise RuntimeError('source changed during run')
        report['numerical_check_failures'] = sum(r['status'] == 'failed_numerical_check' for r in report['records'])
        report['complete'] = True
    finally:
        setNeighborLanes(None)
        save(a.output, report)


if __name__ == '__main__':
    main()
