"""Run the small public examples in one process precision; fail on any check.

Use a fresh invocation for each --precision. No source-path injection: install
the package first. The wheel checker also calls run() with the isolated wheel
already selected, so it executes these same examples without a checkout import.
"""

import argparse
import json
import os
from pathlib import Path
import runpy


def run(device='cpu', precision=None):
    import torch
    from warpSPHCore import get_torch_precision

    dtype = get_torch_precision()
    if precision is not None and dtype != getattr(torch, precision):
        raise RuntimeError('active precision differs from requested precision; check warpSPHCore_config')
    root = Path(__file__).resolve().parent
    records = []

    def example(name, function, **kwargs):
        module = runpy.run_path(str(root / (name+'.py')))
        record = dict(example=name, **module[function](device=device, **kwargs))
        records.append(record)
        print(json.dumps(record), flush=True)

    for name in ('constrained_poisson', 'singular_graph'):
        example(name, 'solve', dtype=dtype)
    example('singular_graph', 'solve', dtype=dtype, projectIncompatible=True)
    example('solver_capabilities', 'saddlePoint', dtype=dtype)
    example('solver_capabilities', 'shiftingFallback', dtype=dtype)
    for side in ('left', 'right'):
        example('vector_gmres', 'solve', dtype=dtype, side=side)
    for crk in (False, True):
        example('corrected_autodiff', 'run', crk=crk)
    example('reconstruct_field', 'run')
    for backend in ('torch', 'warp'):
        for mode in ('MFM', 'MFV'):
            example('mfm_rates', 'run', backend=backend, mode=mode)
    if device == 'cuda':
        torch.cuda.synchronize()
    print(f'Checked {len(records)} example configurations on {device}, {dtype}', flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    args = parser.parse_args()
    os.environ['warpSPHCore_PRECISION'] = args.precision
    run(args.device, args.precision)
