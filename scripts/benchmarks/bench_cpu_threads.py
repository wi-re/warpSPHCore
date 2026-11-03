"""Run the existing qualified backend benchmark with explicit CPU placements."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2],
                    help='Repository root (defaults to the checkout containing this script)')
parser.add_argument('--output-dir', required=True, type=Path)
parser.add_argument('--native-library', required=True, type=Path)
parser.add_argument('--threads', nargs='+', type=int, default=[4, 8, 16, 24])
args = parser.parse_args()
allowed = sorted(os.sched_getaffinity(0))
if any(t < 1 or t > len(allowed) for t in args.threads):
    parser.error('Thread counts must fit the available CPU affinity mask')
args.output_dir.mkdir(parents=True, exist_ok=True)
manifest = dict(complete=False, allowed_cpus=allowed, placement='first T allowed logical CPUs; OMP_PROC_BIND=close; OMP_PLACES=cores', topology=[], runs=[])
for cpu in allowed:
    root = Path(f'/sys/devices/system/cpu/cpu{cpu}')
    def read(name):
        p = root / name
        return p.read_text().strip() if p.exists() else None
    manifest['topology'].append(dict(cpu=cpu, core_id=read('topology/core_id'), max_khz=read('cpufreq/cpuinfo_max_freq'), l2_shared_cpus=read('cache/index2/shared_cpu_list')))
manifest['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest_path = args.output_dir / 'manifest.json'
for threads in args.threads:
    cpus = allowed[:threads]
    output = args.output_dir / f'threads-{threads}.json'
    command = ['taskset', '--cpu-list', ','.join(map(str, cpus)), sys.executable,
               str(args.root / 'scripts/benchmarks/bench_backends.py'),
               '--devices', 'cpu', '--dims', '2', '--ns', '1000000',
               '--threads', str(threads), '--backends', 'warp', 'native_torch', 'standalone', 'standalone_reuse', '--native-library', str(args.native_library),
               '--output', str(output)]
    environment = os.environ | dict(OMP_NUM_THREADS=str(threads), OMP_PROC_BIND='close', OMP_PLACES='cores', MKL_NUM_THREADS=str(threads))
    manifest['runs'].append(dict(threads=threads, cpus=cpus, command=command, environment={k:environment[k] for k in ['OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES', 'MKL_NUM_THREADS']}, output=str(output)))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Starting {threads} workers on CPUs {cpus}', flush=True)
    subprocess.run(command, env=environment, cwd=args.root, check=True)
manifest['complete'] = True
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
records = []
for run in manifest['runs']:
    report = json.loads(Path(run['output']).read_text())
    if not report['complete']:
        raise RuntimeError('An incomplete run cannot be analyzed')
    records.extend(dict(threads=run['threads'], **r) for r in report['records'])
columns = ['threads', 'device', 'dimension', 'n', 'edges', 'operation', 'backend', 'median_ms', 'min_ms', 'max_ms', 'correctness']
with (args.output_dir / 'summary.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(records)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
operations = ['search', 'density', 'interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance']
backends = [('warp', 'Warp (serial)', '#444444', 'o'), ('native_torch', 'Native / Torch', '#0072b2', 's'), ('standalone', 'C++ allocate', '#d55e00', '^'), ('standalone_reuse', 'C++ reuse', '#009e73', 'D')]
fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
for axis, operation in zip(axes.flat, operations):
    for backend, label, color, marker in backends:
        points = sorted((r for r in records if r['operation'] == operation and r['backend'] == backend), key=lambda r:r['threads'])
        if not points:
            continue
        x = [r['threads'] for r in points]
        axis.plot(x, [r['median_ms'] / 1000 for r in points], color=color, marker=marker, label=label, linestyle='--' if backend == 'warp' else '-')
        axis.fill_between(x, [r['min_ms'] / 1000 for r in points], [r['max_ms'] / 1000 for r in points], color=color, alpha=.12)
    axis.set_title(operation.capitalize())
    axis.set_xlabel('Worker count / available CPUs')
    axis.set_ylabel('Completed wall time (s)')
    axis.set_xticks(args.threads)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=.25)
handles, labels = axes.flat[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=4)
fig.suptitle('CPU threading: 1,000,000 particles in 2D, FP64, ≈33 neighbors\nPinned CPU sets; static row partitioning; median and min–max of 3 samples; topology in manifest.json')
fig.savefig(args.output_dir / 'cpu-thread-scaling.png', dpi=170)
plt.close(fig)
print('Thread sweep complete:', args.output_dir, flush=True)
