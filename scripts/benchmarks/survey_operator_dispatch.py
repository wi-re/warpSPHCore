#!/usr/bin/env python3
"""Untimed CPU dispatch census of production wrappers, with frozen inputs.

Counts Torch dispatcher operations and actual Warp launch requests; records no
performance samples. CPU scalar reads identify sites to check on CUDA, not
measured CUDA synchronizations. Vendor CUDA kernel counts need a GPU trace.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from _operator_cases import Case, OPERATIONS, Workload, check_output
from bench_core_operators import save, source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    import warp as wp
    from torch.utils._python_dispatch import TorchDispatchMode
    from warpSPHCore import setNeighborLanes, get_torch_precision
    wp.config.kernel_cache_dir = str(a.output.parent / 'warp-cache')
    wp.init()
    torch.set_num_threads(2)
    setNeighborLanes(1)
    side = 16 if a.dim == 2 else 8
    w = Workload(side ** a.dim, a.dim, 25, 'cpu')
    w.prepared('Wendland2')
    report = dict(complete=False, scope=__doc__, dim=a.dim, dtype=str(get_torch_precision()),
                  torch=torch.__version__, warp=wp.__version__, source_sha256=source_hashes(), records=[])
    for name in OPERATIONS:
        c = Case(w.n, a.dim, 25, name)
        call = w.call(c)
        check_output(call())
        counts, launches = Counter(), []
        class Census(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                counts[str(func)] += 1
                return func(*args, **(kwargs or {}))
        original = wp.launch
        def launch(kernel, dim, *args, **kwargs):
            launches.append(dict(kernel=kernel.key, dim=dim, block_dim=kwargs.get('block_dim', 256)))
            return original(kernel, dim, *args, **kwargs)
        wp.launch = launch
        try:
            with Census():
                output = call()
        finally:
            wp.launch = original
        check_output(output)
        report['records'].append(dict(operation=name, torch_dispatch=dict(counts), warp_launches=launches,
                                       warp_launch_count=len(launches), torch_dispatch_count=sum(counts.values()),
                                       cpu_scalar_reads=sum(v for k, v in counts.items() if '_local_scalar_dense' in k)))
        save(a.output, report)
    if source_hashes() != report['source_sha256']:
        raise RuntimeError('sources changed during dispatch census')
    report['complete'] = True
    save(a.output, report)
    setNeighborLanes(None)


if __name__ == '__main__':
    main()
