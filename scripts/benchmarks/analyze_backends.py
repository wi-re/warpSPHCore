#!/usr/bin/env python3
"""Plot an already-qualified backend sweep and export a compact CSV.

This script imports neither Torch nor Warp and never runs measured workloads.
It accepts only complete reports and preserves every recorded timing sample.
"""
import argparse
import csv
import json
from pathlib import Path

BACKENDS = ['warp', 'native_torch', 'standalone', 'standalone_reuse']
LABELS = ['Warp / Torch', 'Native / Torch', 'C++ allocate', 'C++ reuse']
COLORS = ['#4c78a8', '#f58518', '#e45756', '#54a24b']
OPERATIONS = ['density', 'interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance']


def plot(records, operations, devices, dims, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(device, operation) for device in devices for operation in operations]
    figure, axes = plt.subplots(len(rows), len(dims), squeeze=False, sharey='row',
                                 figsize=(4 * len(dims), 2.9 * len(rows)))
    for row, (device, operation) in enumerate(rows):
        for column, dimension in enumerate(dims):
            axis = axes[row, column]
            for backend, label, color, marker in zip(BACKENDS, LABELS, COLORS, ['o', 's', '^', 'D']):
                series = sorted((record for record in records if record['operation'] == operation
                    and record['device'] == device and record['dimension'] == dimension
                    and record['backend'] == backend), key=lambda record: record['n'])
                if not series:
                    continue
                ns = [record['n'] for record in series]
                axis.plot(ns, [record['median_ms'] * 1000 for record in series],
                          label=label, color=color, marker=marker, markersize=3, linewidth=1.5)
                axis.fill_between(ns, [record['min_ms'] * 1000 for record in series],
                                  [record['max_ms'] * 1000 for record in series], color=color, alpha=.10)
            axis.set(xscale='log', yscale='log', xlabel='Particles', ylabel='Completed wall time (µs)',
                     title=f'{operation.capitalize()} · {device.upper()} · {dimension}D')
            if column:
                axis.set_ylabel('')
            axis.grid(True, which='major', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), frameon=False)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, .035, 1, .975))
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.input]
    if any(not run['complete'] for run in runs):
        parser.error('Input sweep is incomplete; no performance conclusions generated')
    report = runs[0]
    if len(runs) > 1:
        for run in runs[1:]:
            for key in ('precision', 'csr', 'fields', 'operators', 'search'):
                if run['scope'].get(key) != report['scope'].get(key):
                    parser.error(f'Cannot combine different workload contracts: {key}')
            for key in ('torch', 'warp', 'gpu'):
                if run['environment'].get(key) != report['environment'].get(key):
                    parser.error(f'Cannot combine different environments: {key}')
            if run['configuration']['threads'] != report['configuration']['threads']:
                parser.error('Cannot combine different CPU thread counts')
        report = dict(report)
        report['records'] = [record | {'run_index': index,
                                       'warmup_batches': record.get('warmup_batches', run['configuration']['warmup'])}
                             for index, run in enumerate(runs) for record in run['records']]
        keys = [(record['device'], record['dimension'], record['n'],
                 record['operation'], record['backend']) for record in report['records']]
        if len(keys) != len(set(keys)):
            parser.error('Input sweeps contain duplicate timing cases')
        report['configuration'] = dict(report['configuration'])
        for key in ('ns', 'dims', 'devices'):
            report['configuration'][key] = sorted({value for run in runs
                                                   for value in run['configuration'][key]})
        report['activity'] = [item for run in runs for item in run['activity']]
        report['runs'] = [dict(path=str(path), metadata={key: value for key, value in run.items()
                              if key not in ('records', 'activity')})
                          for path, run in zip(args.input, runs)]
        # Per-run source/build provenance remains authoritative after merging.
        report.pop('source_sha256', None)
        report.pop('build_sha256', None)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'combined-results.json').write_text(json.dumps(report, indent=2) + '\n')
    records = report['records']
    if any(record['backend'] not in BACKENDS for record in records):
        parser.error('Legacy verification-mode results require their archived analysis script')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ['device', 'dimension', 'n', 'edges', 'mean_neighbors', 'operation', 'backend',
              'iterations', 'warmup_batches', 'measured_batches', 'median_ms', 'min_ms', 'max_ms', 'median_submitted_ms', 'correctness']
    with (args.output_dir / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(record | {'measured_batches': len(record['completed_ms'])}
                         for record in records)
    devices, dims = report['configuration']['devices'], report['configuration']['dims']
    precision = {'float32': 'FP32', 'float64': 'FP64'}[report['scope']['precision']]
    plot(records, ['search'], devices, dims, args.output_dir / 'search.png',
         f'Fresh compact-hash search · medians and sample ranges · {precision}')
    plot(records, ['interpolate'], devices, dims, args.output_dir / 'interpolate.png',
         f'Interpolation scaling · normal execution · {precision} · fixed CSR')
    plot(records, ['density', 'gradient'], devices, dims, args.output_dir / 'overview.png',
         f'Fixed-CSR SPH · medians and sample ranges · {precision}')
    for device in devices:
        plot(records, OPERATIONS, [device], dims, args.output_dir / f'operators-{device}.png',
             f'All seven SPH primals · {device.upper()} · {precision} · fixed CSR')
    lines = ['# Native backend scaling graphs', '',
             f"{len(records)} qualified timing records. Curves show median completed wall time; bands show sample min/max.", '',
             'Compilation, input generation, correctness checks and CSR canonicalization are excluded.',
             'Every path uses ordinary execution, with no numerical-content or CSR verification.',
             'The smallest cases bound calling overhead and also include computation.', '',
             '## Fresh neighborhood search', '', '![Fresh search](search.png)', '',
             '## Interpolation', '', '![Interpolation](interpolate.png)', '']
    for device in devices:
        lines += [f'## All {device.upper()} operators', '',
                  f'![All {device.upper()} operators](operators-{device}.png)', '']
    lines += ['## Measurement scopes', '']
    for name, description in report['scope'].items():
        lines += [f'- **{name}:** {description}']
    lines += ['', 'Raw samples, source/build hashes and per-case activity snapshots are in the input JSON.',
              'Fresh native GPU allocation uses cudaMalloc/hipMalloc; Torch uses its caching allocator. ',
              'Every timed path uses normal execution without numerical or CSR verification.',
              'Submission intervals include API-internal waits and host gaps; they are not GPU kernel durations.',
              'CPU results compare each implementation at the stated thread setting; Warp CPU parallelism differs.',
              'Large cases use fewer repeated calls; actual batch sizes and samples are retained in JSON/CSV.',
              'AMD runtime performance is unmeasured. Desktop activity and clock variation can affect this initial pass.', '']
    (args.output_dir / 'graphs.md').write_text('\n'.join(lines))
    print(args.output_dir / 'graphs.md')


if __name__ == '__main__':
    main()
