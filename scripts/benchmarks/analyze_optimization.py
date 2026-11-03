#!/usr/bin/env python3
"""Compare complete before/after sweeps without importing Torch or running kernels.

Supply matching report sets through --baseline and --optimized. Every raw case
is exported to CSV; the figure highlights 2D million-particle workloads. Source
reports retain samples, provenance and qualifications, including cutoff rounding.
"""
import argparse
import csv
import json
from pathlib import Path

OPERATIONS = ['search', 'density', 'interpolate', 'gradient', 'divergence', 'curl',
              'laplacian', 'covariance']
KEY = ('device', 'precision', 'dimension', 'n', 'operation', 'backend')


def read(paths):
    records = {}
    for path in paths:
        run = json.loads(path.read_text())
        if not run['complete']:
            raise ValueError(f'Incomplete report: {path}')
        for record in run['records']:
            record = record | {'precision': run['scope']['precision'], 'source_report': str(path)}
            key = tuple(record[name] for name in KEY)
            if key in records:
                raise ValueError(f'Duplicate case: {key}')
            records[key] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', nargs='+', type=Path, required=True)
    parser.add_argument('--optimized', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    before, after = read(args.baseline), read(args.optimized)
    if before.keys() != after.keys():
        parser.error('Before/after reports must contain the same cases')
    rows = []
    for key, result in after.items():
        initial = before[key]
        if initial['edges'] != result['edges']:
            parser.error(f'CSR edge counts changed: {key}')
        rows.append({name: result[name] for name in KEY} | dict(
            edges=result['edges'], baseline_ms=initial['median_ms'],
            optimized_ms=result['median_ms'], speedup=initial['median_ms']/result['median_ms'],
            baseline_min_ms=initial['min_ms'], baseline_max_ms=initial['max_ms'],
            optimized_min_ms=result['min_ms'], optimized_max_ms=result['max_ms'],
            baseline_report=initial['source_report'], optimized_report=result['source_report']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'before-after.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    import numpy as np
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row_index, device in enumerate(['cuda', 'cpu']):
        for column, precision in enumerate(['float32', 'float64']):
            axis = axes[row_index, column]
            cases = [r for r in rows if r['device'] == device and r['precision'] == precision
                     and r['dimension'] == 2 and r['n'] == 1_000_000
                     and r['backend'] == 'native_torch']
            cases = {r['operation']: r for r in cases}
            operations = [op for op in OPERATIONS if op in cases]
            y = np.arange(len(operations))
            for offset, backend, stage, label, color in [
                (-.23, 'native_torch', 'baseline', 'Native before', '#9aabb9'),
                (0, 'native_torch', 'optimized', 'Native optimized', '#e07820'),
                (.23, 'warp', 'optimized', 'Warp reference (same sweep)', '#3977a8')]:
                values, lows, highs = [], [], []
                for op in operations:
                    record = (before if stage == 'baseline' else after)[
                        (device, precision, 2, 1_000_000, op, backend)]
                    values.append(record['median_ms'])
                    lows.append(record['median_ms'] - record['min_ms'])
                    highs.append(record['max_ms'] - record['median_ms'])
                axis.barh(y+offset, values, height=.21, color=color, label=label,
                          xerr=[lows, highs], error_kw=dict(elinewidth=.7, capsize=1.5))
            axis.set_yticks(y, [f"{op.capitalize()}  ({cases[op]['speedup']:.2f}×)"
                               for op in operations])
            axis.invert_yaxis()
            axis.set_xscale('log')
            axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel('Completed wall time (ms, log scale)')
            axis.set_title(f"{'GPU' if device == 'cuda' else 'CPU · 16 workers'} · "
                           f"{'FP32' if precision == 'float32' else 'FP64'}")
            axis.grid(axis='x', alpha=.2)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    figure.suptitle('1,000,000 particles · 2D · ≈33 neighbors per particle\n'
                   'Parentheses: native speedup vs saved baseline; whiskers: sample min–max', fontsize=14)
    figure.savefig(args.output_dir / 'before-after.png', dpi=160)
    plt.close(figure)
    lines = ['# Native optimization comparison', '',
             f'{len(rows)} matched cases. All raw results are in before-after.csv.', '',
             'Medians are completed wall times; bounds are sample min–max, not confidence intervals.',
             'Warp is the contemporary reference from the optimized sweep. CPU Warp launches are serial.',
             'No correctness verification is in timed execution; qualification is retained in source JSON.',
             'Desktop/background activity can perturb individual cases; retain repeats rather than cherry-picking.', '',
             '![2D million-particle comparison](before-after.png)', '']
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines))
    print(args.output_dir / 'comparison.md')


if __name__ == '__main__':
    main()
