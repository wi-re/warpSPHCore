"""Summarize several policy-migration studies without merging their samples."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_production import NAMES, elapsed


def summarize(path):
    report = json.loads(path.read_text())
    assert report['complete'] and all(case['complete'] for case in report['fixtures'])
    pairs = []
    for case in report['fixtures']:
        lookup = {(row['device'], row['operation'], row['path']): row for row in case['records']}
        for device in ('cpu', 'cuda'):
            for operation in NAMES:
                if (device, operation, 'baseline') not in lookup:
                    continue
                old, new = [lookup[device, operation, version] for version in ('baseline', 'candidate')]
                ranges = []
                for record in (old, new):
                    values = [elapsed(sample, device) for sample in record['samples']]
                    ranges.append((max(values) - min(values)) / statistics.median(values))
                pairs.append({**{key: case[key] for key in ('dimension', 'n', 'nh', 'precision', 'wide')},
                    'device': device, 'operation': operation,
                    'baseline_ms': elapsed(old, device), 'candidate_ms': elapsed(new, device),
                    'speedup': elapsed(old, device) / elapsed(new, device),
                    'sample_range_fraction': ranges})
    return dict(source=str(path), kernel=report['kernel'], support=report['support'],
        gradient_scheme=report['gradient_scheme'], fixtures=len(report['fixtures']),
        comparisons=len(pairs), pairs=pairs,
        largest_sampled_output_error=max(check['error'] for case in report['fixtures']
            for check in case['qualification']),
        other_gpu_processes=sum(bool(item['other_processes']) for item in report['activity']),
        metrics={device: dict(
            median_speedup=statistics.median(pair['speedup'] for pair in pairs if pair['device'] == device),
            median_sample_range=statistics.median(value for pair in pairs if pair['device'] == device
                for value in pair['sample_range_fraction']),
            large_regressions=[pair for pair in pairs if pair['device'] == device
                and pair['n'] >= 262144 and pair['speedup'] < .9]) for device in ('cpu', 'cuda')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('studies', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--nh', type=float, choices=[4., 2.57], default=4.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [summarize(path) for path in args.studies]
    figure, axes = plt.subplots(len(reports), 4, figsize=(19, 3.3 * len(reports)), squeeze=False)
    short = ['Density', 'Interp S', 'Interp V', 'Interp M', 'Grad S', 'Grad V', 'Grad M']
    for row, report in enumerate(reports):
        dimension = max(pair['dimension'] for pair in report['pairs'])
        n = max(pair['n'] for pair in report['pairs'] if pair['dimension'] == dimension)
        for column, (device, precision) in enumerate(
                [('cuda', 'float32'), ('cuda', 'float64'), ('cpu', 'float32'), ('cpu', 'float64')]):
            axis = axes[row, column]
            selected = {pair['operation']: pair for pair in report['pairs']
                if pair['dimension'] == dimension and pair['n'] == n and pair['nh'] == args.nh
                and pair['device'] == device and pair['precision'] == precision and not pair['wide']}
            values = [selected[name]['speedup'] for name in NAMES]
            axis.bar(np.arange(len(values)), values,
                color=['#1677b8' if value >= 1 else '#b76b32' for value in values])
            axis.axhline(1, color='#333333', linewidth=1)
            axis.set_ylim(0, max(1.3, max(values) * 1.15))
            axis.set_xticks(np.arange(len(values)), short, rotation=35, ha='right')
            axis.set_title(f'{device.upper()} · {precision.upper()} · {dimension}D · {n:,}')
            axis.grid(axis='y', alpha=.2)
            if column == 0:
                axis.set_ylabel(f"{report['kernel']} / {report['support']}\n"
                    f"{report['gradient_scheme']} gradient\nSpeedup = previous time / migrated time")
    figure.suptitle(f'Shared native policies versus the preceding native implementation · n_h={args.nh:g}\n'
        'Above 1 is faster; S / V / M = scalar / vector / matrix; allocating public calls')
    figure.tight_layout(rect=(0, 0, 1, .95))
    filename = 'policy-expansion.png' if args.nh == 4 else 'policy-expansion-nh2.57.png'
    figure.savefig(args.output/filename, dpi=165)
    plt.close(figure)
    (args.output/'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps([{key: value for key, value in report.items() if key != 'pairs'}
        for report in reports], indent=2))


if __name__ == '__main__':
    main()
