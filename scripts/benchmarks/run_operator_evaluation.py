#!/usr/bin/env python3
"""Run an operator evaluation phase in separate precision/dimension processes.

Defaults to writing plans. Performance timing is always an explicit phase.
No CUDA code is imported by this driver. Outputs remain incremental on failure.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def commands(a):
    for precision in a.precisions:
        for dim in a.dims:
            env = dict(os.environ, warpSPHCore_PRECISION=precision)
            name = f'{a.stage}-{dim}d-{precision}'
            cmd = [sys.executable]
            if a.stage == 'resources':
                cmd += [str(HERE / 'survey_operator_resources.py'), '--dim', str(dim), '--lanes', '1', '4', '8', '16', '32']
                if a.ptxas:
                    cmd += ['--ptxas', a.ptxas]
            else:
                sizes = ([1024] if dim == 2 else [512]) if a.stage == 'validate' else (
                    [1024, 4096, 16384, 65536, 262144, 1048576] if dim == 2 else [512, 4096, 32768, 262144, 2097152])
                cmd += [str(HERE / 'bench_core_operators.py'), a.stage, '--dim', str(dim), '--ns', *map(str, sizes)]
                if a.stage == 'validate':
                    cmd += ['--lanes', '1', '--forms', 'plain', 'renorm', 'crk']
                elif a.stage == 'benchmark':
                    cmd += ['--device', a.device, '--keep-going', '--max-estimated-call-ms', str(a.max_estimated_call_ms)]
            cmd += ['--output', str(a.output_dir / f'{name}.json')]
            yield env, cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['plan', 'validate', 'resources', 'benchmark'], default='plan')
    p.add_argument('--precisions', choices=['float32', 'float64'], nargs='+', default=['float32', 'float64'])
    p.add_argument('--dims', choices=[2, 3], type=int, nargs='+', default=[2, 3])
    p.add_argument('--max-estimated-call-ms', type=float, default=1000., help='budget for projected larger per-call times; skipped cases remain explicit')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ptxas')
    p.add_argument('--output-dir', required=True, type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.stage == 'benchmark':
        # Fail before touching CUDA if prerequisite correctness evidence is absent
        # or predates changes to the measured source/suite.
        import json
        from bench_core_operators import source_hashes
        for precision in a.precisions:
            for dim in a.dims:
                prerequisite = a.output_dir / f'validate-{dim}d-{precision}.json'
                doc = json.loads(prerequisite.read_text())
                if not doc.get('complete') or any(r.get('status') != 'qualified' for r in doc.get('records', [])) or doc.get('source_sha256') != source_hashes():
                    p.error(f'missing/current qualification required: {prerequisite}')
    for env, command in commands(a):
        print(' '.join(command), flush=True)
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
