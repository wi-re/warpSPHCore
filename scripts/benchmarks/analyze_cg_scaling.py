#!/usr/bin/env python3
"""Summarize saved scaling sweeps and render a static scientific figure.

The empirical knee is the first sampled N with T >= 2*T(N_min), sustained at
all larger sampled sizes. It is a bracket on this grid, not a fitted claim
that exactly half the time is launch overhead. Adjacent log-log slopes expose
the transition and cache/throughput changes without assuming an affine model.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from _work_metrics import cgWork

LABELS = {'torch': 'Torch eager', 'native': 'Core native',
          'native-cached': 'Core cached', 'warp-cached': 'Warp conditional'}


def describe(points):
    points = sorted(points)
    baseline = points[0][1]
    knee = next((i for i, (_, t) in enumerate(points)
                 if t >= 2 * baseline and all(v >= 2 * baseline for _, v in points[i:])), None)
    return dict(smallest_n=points[0][0], floor_ms=baseline, largest_n=points[-1][0],
                largest_ms=points[-1][1], largest_over_floor=points[-1][1] / baseline,
                empirical_2x_knee=None if knee is None else dict(
                    lower_exclusive=points[knee - 1][0] if knee else 0,
                    upper_inclusive=points[knee][0]),
                adjacent_elasticities=[dict(n_low=a, n_high=b, exponent=math.log(tb / ta) / math.log(b / a))
                                       for (a, ta), (b, tb) in zip(points, points[1:])],
                points=[dict(n=n, ms=t) for n, t in points])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    parser.add_argument('--throughput-plot', type=Path)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if not raw['complete']:
        parser.error('input sweep must be complete')
    groups = defaultdict(list)
    byKey = {}
    for case in raw['cases']:
        result = case['result']
        row = result['records'][0]
        degree = row['neighborhood']['mean'] if row['neighborhood'] else None
        groups[(result['workload'], degree, result['engine'])].append(row)
        byKey[(result['workload'], degree, row['n'], result['engine'])] = row
    summary = dict(source=str(args.input), complete=True,
                   input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   analysis_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__), Path(__file__).with_name('_work_metrics.py'))},
                   knee_definition=__doc__.split('\n\n', 1)[1].strip(), solves=[], isolated_matvecs=[], work_ledgers=[],
                   validation=dict(cases=len(raw['cases']), matvec_output_parity=True,
                                   fixed_scheduled_work=True, fixed_matvec_work=True,
                                   symmetry_max=0., sample_relative_iqr_max=0.))
    for (workload, degree, engine), rows in groups.items():
        points = [(r['n'], r['median_wall_ms']) for r in rows]
        summary['solves'].append(dict(workload=workload, degree=degree, engine=engine, **describe(points)))
        for row in rows:
            if workload == 'sph':
                summary['work_ledgers'].append(dict(workload=workload, degree=degree, engine=engine, n=row['n'],
                                                    completed_updates=row['completed_updates'],
                                                    scheduled_updates=row['scheduled_updates'],
                                                    work=cgWork(row, engine=engine)))
            validation = summary['validation']
            validation['fixed_scheduled_work'] &= row['scheduled_updates'] == row['iterations']
            validation['fixed_matvec_work'] &= all(v == row['iterations'] + 2 for v in row['matvecs'])
            validation['symmetry_max'] = max(validation['symmetry_max'], row['symmetry_probe'])
            quantiles = statistics.quantiles(row['wall_ms'], n=4)
            validation['sample_relative_iqr_max'] = max(validation['sample_relative_iqr_max'],
                                                      (quantiles[2] - quantiles[0]) / row['median_wall_ms'])
            if row['isolated_matvec']:
                validation['matvec_output_parity'] &= row['isolated_matvec']['matches_eager_output']
        if engine == 'native-cached' and all(r['isolated_matvec'] for r in rows):
            points = [(r['n'], r['isolated_matvec']['median_graph_device_ms_per_call']) for r in rows]
            largest = max(rows, key=lambda r: r['n'])
            metric = dict(workload=workload, degree=degree, **describe(points))
            if degree:
                metric['largest_ns_per_csr_entry'] = metric['largest_ms'] * 1e6 / largest['neighborhood']['edges']
            summary['isolated_matvecs'].append(metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    if not args.plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = dict(zip(LABELS, ['#686868', '#2a78b7', '#d35d22', '#33976f']))
    neighborColors = ['#386cb0', '#f28e2b', '#59a14f', '#af5ba7']
    degrees = sorted({d for workload, d, _ in groups if workload == 'sph'})
    for engine in LABELS:
        rows = sorted(groups.get(('stencil', None, engine), []), key=lambda r: r['n'])
        if rows:
            axes[0, 0].loglog([r['n'] for r in rows], [r['median_wall_ms'] for r in rows],
                              'o-', color=colors[engine], label=LABELS[engine], markersize=4)
    axes[0, 0].set(title=f"Stencil: {raw['configuration']['stencil_iterations']} fixed CG updates", ylabel='Solve wall time (ms)')
    for degree, color in zip(degrees, neighborColors):
        rows = sorted(groups.get(('sph', degree, 'native-cached'), []), key=lambda r: r['n'])
        if not rows:
            continue
        axes[0, 1].loglog([r['n'] for r in rows], [r['median_wall_ms'] for r in rows],
                          'o-', color=color, label=f'{degree:g} neighbors', markersize=4)
        axes[1, 0].loglog([r['n'] for r in rows],
                          [r['isolated_matvec']['median_graph_device_ms_per_call'] for r in rows],
                          'o-', color=color, label=f'{degree:g} neighbors', markersize=4)
    sphIterations = raw['configuration']['sph_iterations']
    axes[0, 1].set(title=f'SPH: core cached, {sphIterations} fixed updates', ylabel='Solve wall time (ms)')
    axes[1, 0].set(title='SPH action: warmed batch graph', ylabel='Device interval per matvec (ms)')
    if degrees:
        degree = min(degrees, key=lambda d: abs(d - 55))
        for engine in ('torch', 'native', 'warp-cached'):
            rows = sorted(groups.get(('sph', degree, engine), []), key=lambda r: r['n'])
            rows = [r for r in rows if ('sph', degree, r['n'], 'native-cached') in byKey]
            axes[1, 1].semilogx([r['n'] for r in rows],
                                [r['median_wall_ms'] / byKey[('sph', degree, r['n'], 'native-cached')]['median_wall_ms']
                                 for r in rows], 'o-', color=colors[engine], label=LABELS[engine], markersize=4)
        axes[1, 1].axhline(1, color='#999999', linestyle='--', linewidth=1)
        axes[1, 1].set(title=f'SPH, {degree:g} neighbors: schedule cost converges',
                       ylabel='Solve time / core cached solve time')
    for ax in axes.flat:
        ax.set_xlabel('Particles N')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda n, _: f'{n / 1e6:g}M' if n >= 1e6 else f'{n / 1e3:g}k'))
        ax.grid(True, which='major', alpha=.23)
        ax.legend(fontsize=9)
    first = raw['cases'][0]['result']
    fig.suptitle(f"CG size and sparsity scaling | {first['dtype'].removeprefix('torch.')} | {first['repeat']} samples, setup excluded\n"
                 'RTX PRO 6000 Blackwell; fixed 2D periodic adjacency; equal work, not equal accuracy', fontsize=13)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)
    if args.throughput_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
        if degrees:
            degree = min(degrees, key=lambda d: abs(d - 55))
            for engine in LABELS:
                ledgers = sorted((w for w in summary['work_ledgers'] if w['degree'] == degree and w['engine'] == engine),
                                 key=lambda w: w['n'])
                axes[0].semilogx([w['n'] for w in ledgers],
                                 [w['work']['directed_interactions_per_second'] / 1e9 for w in ledgers],
                                 'o-', label=LABELS[engine], color=colors[engine], markersize=4)
            axes[0].set(title=f'Whole CG call, {degree:g} neighbors', ylabel='Billion directed interactions / wall second')
            for degree, color in zip(degrees, neighborColors):
                ledgers = sorted((w for w in summary['work_ledgers'] if w['degree'] == degree and w['engine'] == 'native-cached'),
                                 key=lambda w: w['n'])
                axes[1].semilogx([w['n'] for w in ledgers],
                                 [w['work']['isolated_matvec']['directed_interactions_per_second'] / 1e9 for w in ledgers],
                                 'o-', label=f'{degree:g} neighbors', color=color, markersize=4)
            axes[1].set(title='Isolated SPH action, warmed graph', ylabel='Billion directed interactions / device second')
        for ax in axes:
            ax.set_xlabel('Particles N')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda n, _: f'{n / 1e6:g}M' if n >= 1e6 else f'{n / 1e3:g}k'))
            ax.grid(True, which='major', alpha=.23)
            ax.legend(fontsize=9)
        fig.suptitle('Interaction throughput: actual non-self CSR pairs × executed matvec passes\n'
                     'Directed pairs; verification included in whole solves; operator and precision held fixed', fontsize=12)
        args.throughput_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.throughput_plot, dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
