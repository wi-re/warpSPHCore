#!/usr/bin/env python3
"""Compile production overloads to PTX and assemble with ptxas, without GPU work.

A tiny CPU fixture specializes generic kernels. wp.launch is intercepted during
inspection, so tiled CUDA kernels never execute. The emitted PTX retains runtime
CSR/grid selection; resources therefore cover both paths, not a pruned branch.
Requires Warp compile_aot_module and a ptxas supporting the target architecture.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from _operator_cases import CORE, LOCAL, Case, Workload
from bench_core_operators import save, source_hashes


def parse_ptxas(log):
    """Preserve zero spill counts; missing compiler statistics stay unavailable."""
    records = {}
    current = None
    for line in log.splitlines():
        match = re.search(r"Compiling entry function '([^']+)'", line)
        if match:
            current = records.setdefault(match[1], {})
        # ptxas also emits properties for called device functions after an
        # entry. Their zero-spill records must not overwrite the entry's data.
        properties = re.search(r'Function properties for (\S+)', line)
        if properties:
            current = records.get(properties[1])
        if current is None:
            continue
        match = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
        if match:
            current.update(zip(('stack_frame_bytes', 'spill_store_bytes', 'spill_load_bytes'), map(int, match.groups())))
        match = re.search(r'Used (\d+) registers', line)
        if match:
            current['registers_per_thread'] = int(match[1])
            stack = re.search(r'(\d+) bytes cumulative stack size', line)
            if stack:
                current['cumulative_stack_bytes'] = int(stack[1])
            shared = re.search(r'(\d+) bytes smem', line)
            current['static_shared_bytes'] = int(shared[1]) if shared else 0
    return records


def sm120_upper_bound(registers, block_dim):
    """Optimistic ceiling, NOT CUDA's allocation-granular occupancy calculator.

    48 warps, 32 blocks, 65536 registers/SM per NVIDIA Blackwell tuning guide.
    Omits allocation granularities, shared memory, partitions, grid underfill,
    and scheduling. Small blocks still allocate a whole physical warp.
    """
    if registers <= 0 or block_dim <= 0:
        raise ValueError('positive register count and block size required')
    warps = math.ceil(block_dim / 32)
    limits = dict(block_slots=32, warp_slots=48 // warps, registers=65536 // (registers * 32 * warps))
    blocks = min(limits.values())
    return dict(resident_blocks_upper_bound=blocks, active_warps_upper_bound=blocks * warps,
                physical_warp_occupancy_upper_bound=blocks * warps / 48,
                useful_thread_occupancy_upper_bound=blocks * block_dim / (48 * 32), limits=limits,
                scope='optimistic SM120 resource ceiling; ignores shared memory, allocation rounding and grid size; not achieved occupancy')


@contextmanager
def capture_launches(lanes=1):
    import warp as wp
    launches = []
    original = wp.launch
    def record(kernel, dim, inputs=None, outputs=None, **kw):
        # Public dispatch intentionally disables tiled execution on CPU. Select
        # its matching production CUDA variant explicitly for offline inspection.
        tiled = kernel.module.kernels.get(kernel.key + 'Tiled') if lanes > 1 else None
        if tiled is not None:
            kernel = tiled
            dim = [dim, lanes]
            kw['block_dim'] = lanes
        args = list(inputs or []) + list(outputs or [])
        if kernel.is_generic:
            kernel = kernel.add_overload(kernel.infer_argument_types(args))
        shape = [dim] if isinstance(dim, int) else list(dim)
        block = kw.get('block_dim', kernel.module.options.get('block_dim', 256))
        launches.append((kernel, block, shape))
        return None
    wp.launch = record
    try:
        yield launches
    finally:
        wp.launch = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--arch', type=int, default=120, help='explicit compilation target, not detected hardware')
    p.add_argument('--lanes', nargs='+', type=int, choices=[1, 2, 4, 8, 16, 32], default=[1, 4, 8, 16, 32])
    p.add_argument('--operations', nargs='+', choices=CORE + ('crk_volume', 'crk_moments', 'crk_density') + LOCAL,
                   default=list(CORE + ('crk_volume', 'crk_moments', 'crk_density', 'pinv_symmetric', 'apply_pinv', 'pinv_policy')))
    p.add_argument('--forms', nargs='+', choices=['plain', 'renorm', 'crk'], default=['plain'])
    p.add_argument('--ptxas', default=shutil.which('ptxas'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.ptxas:
        p.error('ptxas must be on PATH or supplied via --ptxas')
    import torch
    import warp as wp
    from warpSPHCore import setNeighborLanes
    wp.config.kernel_cache_dir = str(a.output.parent / 'warp-cache')
    wp.init()
    torch.set_num_threads(2)
    report = dict(schema_version=1, mode='offline_compile', complete=False, gpu_kernels_executed=0,
                  architecture=a.arch, dim=a.dim, dtype=str(__import__('warpSPHCore').get_torch_precision()),
                  torch=torch.__version__, warp=wp.__version__, source_sha256=source_hashes(),
                  compiler=subprocess.run([a.ptxas, '--version'], capture_output=True, text=True, check=True).stdout,
                  occupancy_reference='https://docs.nvidia.com/cuda/blackwell-tuning-guide/', records=[], command=sys.argv)
    save(a.output, report)
    side = 16 if a.dim == 2 else 8
    w = Workload(side ** a.dim, a.dim, 25, 'cpu')
    w.prepared('Wendland2')
    captured = []
    for op in a.operations:
        for form in a.forms:
            if form != 'plain' and op not in ('gradient', 'divergence', 'curl', 'laplacian'):
                continue
            for lanes in ([1] if op in LOCAL else a.lanes):
                c = Case(w.n, a.dim, 25, op, lanes=lanes, form=form)
                call = w.call(c)
                setNeighborLanes(lanes)
                with capture_launches(lanes) as launches:
                    call()
                for kernel, block, shape in launches:
                    captured.append((c, kernel, block, shape))
                if not launches:
                    report['records'].append(dict(**c.dictionary(), status='torch_backend',
                                                  note='no Warp launch; requires CUDA profiling for vendor/Torch kernel resources'))
    builds = {}
    try:
        for c, kernel, block, shape in captured:
            key = (kernel.module.name, block)
            if key not in builds:
                directory = a.output.parent / 'compiled' / f'd{a.dim}-{kernel.module.name}-b{block}'
                directory.mkdir(parents=True, exist_ok=True)
                options = dict(kernel.module.options)
                try:
                    wp.set_module_options({'block_dim': block}, module=kernel.module)
                    artifacts = wp.compile_aot_module(kernel.module, arch=a.arch, module_dir=str(directory), use_ptx=True)
                finally:
                    wp.set_module_options(options, module=kernel.module)
                ptx = next(Path(path) for path in artifacts if str(path).endswith('.ptx'))
                # NVRTC's PTX includes a terminal NUL; standalone ptxas expects text.
                assembly_input = ptx.with_suffix('.assemble.ptx')
                assembly_input.write_bytes(ptx.read_bytes().rstrip(b'\x00'))
                binary = ptx.with_suffix('.cubin')
                command = [a.ptxas, '-v', f'-arch=sm_{a.arch}', str(assembly_input), '-o', str(binary)]
                result = subprocess.run(command, text=True, capture_output=True)
                log = result.stdout + result.stderr
                ptx.with_suffix('.ptxas.log').write_text(log)
                if result.returncode:
                    raise RuntimeError(f'ptxas failed; see {ptx.with_suffix(".ptxas.log")}')
                builds[key] = (parse_ptxas(log), ptx, binary)
            entries, ptx, binary = builds[key]
            # Compiling further block variants can change a generic kernel's hash;
            # names in the current artifact are authoritative. The base key is unique.
            matches = [(name, stats) for name, stats in entries.items()
                       if name.startswith(kernel.key + '_') and name.endswith('_cuda_kernel_forward')]
            if len(matches) != 1:
                raise RuntimeError(f'expected one forward entry for {kernel.key}, found {[x[0] for x in matches]}')
            entry, stats = matches[0]
            row = dict(**c.dictionary(), kernel_name=kernel.key, entry=entry, block_dim=block, fixture_launch_shape=shape,
                       status='compiled', **stats, ptx=str(ptx.resolve()), cubin=str(binary.resolve()),
                       ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest(),
                       traversal_scope='runtime CSR/grid branch; resource allocation shared by both')
            if a.arch == 120:
                row['occupancy_ceiling'] = sm120_upper_bound(stats['registers_per_thread'], block)
            report['records'].append(row)
            save(a.output, report)
            print(f'{c.operation}/{c.form}/L{c.lanes} block={block}: {stats}', flush=True)
        if source_hashes() != report['source_sha256']:
            raise RuntimeError('sources changed during resource survey')
        report['complete'] = True
    finally:
        setNeighborLanes(None)
        save(a.output, report)


if __name__ == '__main__':
    main()
