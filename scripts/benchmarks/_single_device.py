"""Timing and provenance shared by source-tree benchmark entry points.

No torch/device initialization at import. CUDA events measure a stream interval,
including idle launch gaps; they are deliberately not called kernel time.
"""
import importlib.metadata
import importlib.util
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def git_info(root):
    def git(*args):
        p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    digest = hashlib.sha256()
    files = sorted((Path(root)/'src').rglob('*.py'))
    files += sorted((Path(root)/'scripts/benchmarks').glob('*.py'))
    files += sorted((Path(root)/'benchmarks/performance').glob('*.py'))
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return dict(root=str(root), revision=git('rev-parse', 'HEAD'), source_sha256=digest.hexdigest(),
                dirty=bool(git('status', '--porcelain')),
                branch=git('branch', '--show-current'))


def metadata(device):
    import torch
    import warpSPHCore
    packages = {}
    source_versions = {}
    repositories = {}
    for name in ('warpSPHCore', 'warpSPH', 'warpSPHIntegrators', 'warpSPHPlotting', 'warp-lang', 'torch'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        if name.startswith('warpSPH'):
            spec = importlib.util.find_spec(name)
            if spec and spec.origin:
                root = Path(spec.origin).resolve().parents[2]
                repositories[name] = git_info(root)
                for node in ast.parse(Path(spec.origin).read_text()).body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
                        if isinstance(node.value, ast.Constant):
                            source_versions[name] = node.value.value
    cpu_model = platform.processor()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu_model = line.split(':', 1)[1].strip()
                break
    result = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                  cpu_model=cpu_model, cpu_count=os.cpu_count(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  packages=packages, source_versions=source_versions,
                  version_note='packages are installed distribution metadata; source_versions are literal source declarations (editable installs can differ)',
                  torch_runtime_version=str(torch.__version__), repositories=repositories, device=str(device),
                  backend=getattr(warpSPHCore, 'backend', 'warp'), cuda_version=torch.version.cuda,
                  environment={k: v for k, v in os.environ.items() if k.startswith(('WARPSPH', 'warpSPHCore_', 'OMP_', 'MKL_'))},
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if torch.device(device).type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        result['gpu'] = dict(name=p.name, total_memory=p.total_memory, multiprocessors=p.multi_processor_count,
                             capability=[p.major, p.minor], uuid=str(getattr(p, 'uuid', 'unknown')))
        try:
            q = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu',
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
            result['nvidia_smi'] = q.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return result


def configure(request):
    # Must run before importing core: its precision is fixed for process lifetime.
    os.environ['warpSPHCore_PRECISION'] = request['precision']
    os.environ['WARPSPHCORE_BACKEND'] = request['backend']
    os.environ['warpSPHCore_DIM'] = 'Any'
    os.environ['warpSPHCore_INDEX_DTYPE'] = request.get('index_dtype', 'int32')
    os.environ['warpSPHCore_OFFSET_DTYPE'] = request.get('offset_dtype', 'int64')
    os.environ['WARPSPHCORE_PROFILING'] = '0'  # Only enabled for the separate instrumented pass.
    os.environ['WARPSPHCORE_AUDIT'] = '0'
    os.environ['OMP_NUM_THREADS'] = str(request['threads'])
    os.environ['MKL_NUM_THREADS'] = str(request['threads'])
    import torch
    torch.set_num_threads(request['threads'])
    torch.set_num_interop_threads(1)
    import warpSPHCore
    actual = getattr(warpSPHCore, 'backend', 'warp')
    if actual != request['backend']:
        raise NotImplementedError(f"requested {request['backend']} backend, imported {actual}; no fallback allowed")
    if actual == 'warp':
        import warp
        warp.init()
    device = torch.device(request['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
    torch.manual_seed(173)
    return device


def synchronize(device):
    import torch
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def summarize(samples):
    if not samples:
        return None
    return dict(median=statistics.median(samples), min=min(samples), max=max(samples),
                mean=statistics.mean(samples), stdev=statistics.stdev(samples) if len(samples) > 1 else 0.)


def measure(fn, device, *, warmup, repeats, calls, validate):
    import torch
    for _ in range(warmup):
        output = fn()
        del output
    synchronize(device)
    output = fn()
    validate(output)  # Qualification and all host reads outside timed windows.
    del output
    synchronize(device)
    cuda = torch.device(device).type == 'cuda'
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(repeats)] if cuda else []
    # Materialize events outside the window (CUDA events are initialized lazily).
    for start, end in events:
        start.record(); end.record()
    synchronize(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for i in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        if cuda:
            events[i][0].record()
        for _ in range(calls):
            output = fn()
            del output  # Include the ordinary output allocation/release lifecycle.
        if cuda:
            events[i][1].record()
        submitted = time.perf_counter()
        synchronize(device)
        t1 = time.perf_counter()
        samples.append(dict(wall_ms=(t1-t0)*1000/calls, submission_ms=(submitted-t0)*1000/calls,
                            stream_ms=events[i][0].elapsed_time(events[i][1])/calls if cuda else None))
    peak_allocated = torch.cuda.max_memory_allocated(device) if cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if cuda else None
    output = fn()
    validate(output)
    del output
    return dict(samples=samples, wall_ms=summarize([s['wall_ms'] for s in samples]),
                stream_ms=summarize([s['stream_ms'] for s in samples]) if cuda else None,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved)


def profile_summary(prof, path):
    prof.export_chrome_trace(str(path))
    syncs = []
    for event in prof.key_averages():
        if any(s in event.key.lower() for s in ('synchronize', 'streamwait', 'eventquery')):
            syncs.append(dict(name=event.key, count=event.count, cpu_total_us=event.cpu_time_total,
                              cpu_self_us=event.self_cpu_time_total))
    return dict(trace=str(path), synchronization_api=syncs,
                note='Separate instrumented pass; durations can overlap. Includes benchmark end synchronization.')


def profile_call(fn, device, path):
    import torch
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    import warpSPHCore.profiling as hooks
    hooks._torchRecordFunction = torch.profiler.record_function
    synchronize(device)
    with torch.profiler.profile(activities=activities) as prof:
        with torch.profiler.record_function('benchmark_workload'):
            fn()
        synchronize(device)
    return profile_summary(prof, path)


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
