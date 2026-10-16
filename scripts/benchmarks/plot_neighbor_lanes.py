#!/usr/bin/env python3
"""Plot paired timing/interaction rates and summarize a saved lane sweep."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_work_scaling import format_axis, point, series


STYLES = {1: ('#666666', 'o'), 2: ('#537a83', 'P'), 4: ('#2878b5', 's'),
          8: ('#299464', '^'), 16: ('#d65b24', 'D'), 32: ('#a45ca6', 'v')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--degree', type=float, default=57)
    parser.add_argument('--plot', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    if not report['complete']:
        parser.error('input sweep must be complete')
    groups = defaultdict(list)
    for row in report['records']:
        groups[(row['traversal'], row['operation'], row['degree'], row['n'])].append(row)
    summary = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   plot_helper_sha256=hashlib.sha256(Path(__file__).with_name('plot_work_scaling.py').read_bytes()).hexdigest(),
                   comparison='same-process serial baseline; best measured lane is not an automatic dispatch policy',
                   max_relative_serial_difference=max(row['relative_max_serial_difference'] for row in report['records']),
                   cases=len(report['records']), points=[])
    for key, rows in groups.items():
        serial = next((row for row in rows if row['lanes'] == 1), None)
        if serial is None:
            raise ValueError('each case requires a serial baseline')
        best = min(rows, key=lambda row: row['isolated']['median_graph_device_ms_per_call'])
        wallBest = min(rows, key=lambda row: row['median_wall_ms'])
        summary['points'].append(dict(traversal=key[0], operation=key[1], degree=key[2], n=key[3],
                                      best_graph_lanes=best['lanes'], best_wall_lanes=wallBest['lanes'],
                                      serial_graph_ms=serial['isolated']['median_graph_device_ms_per_call'],
                                      best_graph_ms=best['isolated']['median_graph_device_ms_per_call'],
                                      graph_speedup=serial['isolated']['median_graph_device_ms_per_call'] / best['isolated']['median_graph_device_ms_per_call'],
                                      serial_wall_ms=serial['median_wall_ms'], best_wall_ms=wallBest['median_wall_ms']))
    traversals = {row['traversal'] for row in report['records']}
    if len(traversals) != 1:
        parser.error('plot one traversal per figure')
    traversal = next(iter(traversals))
    if args.degree not in {row['degree'] for row in report['records']}:
        parser.error('degree was not measured')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for index, operation in enumerate(('laplacian', 'curl')):
        for lanes in sorted({row['lanes'] for row in report['records']}):
            rows = [row for row in report['records'] if row['operation'] == operation
                    and row['degree'] == args.degree and row['lanes'] == lanes]
            points = []
            for row in rows:
                p = point(row, interactions=row['edges'] - row['n'], isolated=True)
                points.append(p)
            color, marker = STYLES[lanes]
            label = 'Serial (1 lane)' if lanes == 1 else f'{lanes} lanes'
            for ax, metric in zip(axes[index], ('time', 'interaction')):
                series(ax, points, metric=metric, label=label, color=color, marker=marker)
        axes[index, 0].set_title(f'{operation.capitalize()}: device interval per call')
        axes[index, 1].set_title(f'{operation.capitalize()}: interaction throughput')
        for ax, metric in zip(axes[index], ('time', 'interaction')):
            format_axis(ax, metric=metric)
            ax.set_xticks(sorted({row['n'] for row in report['records'] if row['degree'] == args.degree}))
            ax.set_autoscaley_on(True)
            ax.autoscale_view(scalex=False, scaley=True)
            if metric == 'interaction':
                ax.set_ylim(bottom=0)
            low, high = ax.get_ylim()
            assert all(low <= y <= high for line in ax.lines for y in line.get_ydata())
        axes[index, 0].legend(fontsize=9, ncol=2)
    fig.suptitle(f'Trial tiled operators: {args.degree:g} neighbors, {traversal} traversal', fontsize=14)
    fig.text(.5, .917, f'{report["gpu"].removeprefix("NVIDIA ").removesuffix(" Workstation Edition")} · {report["dtype"].removeprefix("torch.")} · {report["configuration"]["dim"]}D periodic lattice',
             ha='center', fontsize=9, color='#555555')
    fig.text(.5, .015, f'Warmed {report["configuration"]["batch"]}-call graph · {report["configuration"]["repeat"]} samples · median and shaded IQR.\n'
             'Same math and fixed geometry; one directed non-self pair per interaction. Explicit lane overrides; production defaults retained.',
             ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94), h_pad=2.4, w_pad=2.)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    fig.savefig(args.plot.with_suffix('.svg'))
    plt.close(fig)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved {args.plot}; summarized {len(groups)} size/degree/operator cases')


if __name__ == '__main__':
    main()
