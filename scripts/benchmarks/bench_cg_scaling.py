#!/usr/bin/env python3
"""Run serial, process-isolated CG size/sparsity sweeps, saving every case.

Each child is a fresh CUDA process to release graph pools and adjacency storage
between cases. This driver never times the process or neighbor construction as
part of a steady solve. Use the same Python environment as bench_krylov.py.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]


def sourceHashes():
    paths = sorted((ROOT / 'src/warpSPHCore').rglob('*.py'))
    paths += sorted(Path(__file__).parent.glob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workloads', nargs='+', choices=['stencil', 'sph'], default=['stencil', 'sph'])
    parser.add_argument('--engines', nargs='+', choices=['torch', 'native', 'native-cached', 'warp-cached'],
                        default=['torch', 'native', 'native-cached', 'warp-cached'])
    parser.add_argument('--stencil-ns', nargs='+', type=int,
                        default=[1024, 16384, 65536, 262144, 1048576, 4194304, 16777216])
    parser.add_argument('--sph-ns', nargs='+', type=int,
                        default=[1024, 16384, 65536, 262144, 1048576, 4194304])
    parser.add_argument('--neighbors', nargs='+', type=int, default=[25, 55, 110, 220])
    parser.add_argument('--stencil-iterations', type=int, default=30)
    parser.add_argument('--sph-iterations', type=int, default=8)
    parser.add_argument('--sph-rtol', type=float, default=0)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--matvec-batch', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('output', 'resume')}
    hashes = sourceHashes()
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), configuration=config,
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  dirty_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                  source_sha256=hashes, cases=[], failures=[], complete=False,
                  timing_scope='steady solve includes final verification and observation; neighbor build and capture excluded',
                  process_scope='one fresh child per case; cases run serially on one nondefault Torch CUDA stream')
    if args.resume and output.exists():
        report = json.loads(output.read_text())
        if report['configuration'] != config or report['source_sha256'] != hashes:
            parser.error('resume requires identical configuration and source hashes')
        report['failures'] = []
        report['complete'] = False
    done = {case['key'] for case in report['cases']}
    cases = []
    for workload in args.workloads:
        sizes = args.stencil_ns if workload == 'stencil' else args.sph_ns
        for neighbors in ([None] if workload == 'stencil' else args.neighbors):
            for n in sizes:
                for engine in args.engines:
                    cases.append((workload, neighbors, n, engine))
    for workload, neighbors, n, engine in cases:
        key = f'{workload}/{neighbors}/{n}/{engine}'
        if key in done:
            continue
        with tempfile.TemporaryDirectory(prefix='cg-scaling-') as directory:
            childOutput = Path(directory) / 'result.json'
            command = [sys.executable, str(Path(__file__).with_name('bench_krylov.py')),
                       '--workload', workload, '--engine', engine, '--ns', str(n),
                       '--iterations', str(args.stencil_iterations if workload == 'stencil' else args.sph_iterations),
                       '--rtol', str(0 if workload == 'stencil' else args.sph_rtol),
                       '--repeat', str(args.repeat), '--warmup', str(args.warmup),
                       '--matvec-batch', str(args.matvec_batch), '--output', str(childOutput)]
            if neighbors is not None:
                command += ['--neighbors', str(neighbors)]
            start = time.perf_counter()
            child = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - start
            if child.returncode == 0 and childOutput.exists():
                result = json.loads(childOutput.read_text())
                report['cases'].append(dict(key=key, command=command, process_seconds=elapsed, result=result))
                row = result['records'][0]
                print(f"{len(report['cases'])}/{len(cases)} {key}: {row['median_wall_ms']:.4f} ms, "
                      f"matvec graph {row['isolated_matvec']['median_graph_device_ms_per_call']:.4f} ms"
                      if row['isolated_matvec'] else f"{key}: {row['median_wall_ms']:.4f} ms", flush=True)
            else:
                report['failures'].append(dict(key=key, command=command, returncode=child.returncode,
                                               output_tail=child.stdout[-12000:]))
                print(f'FAILED {key}: {child.stdout[-1500:]}', flush=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['complete'] = len(report['cases']) == len(cases) and not report['failures']
    if sourceHashes() != hashes:
        report['complete'] = False
        report['source_changed_during_run'] = True
    output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['complete']:
        raise SystemExit('scaling sweep incomplete; see saved failures/source metadata')


if __name__ == '__main__':
    main()
