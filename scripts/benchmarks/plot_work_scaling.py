#!/usr/bin/env python3
"""Render paired timing/work-throughput figures from saved benchmark samples.

CPU-only; never reruns benchmarks. PNGs serve repository docs, SVGs serve
export. A manifest pins inputs/source and every plotted count/median/IQR.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from development_paths import coreDevelopmentPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENGINE_STYLE = {
    'torch': ('Torch eager', '#666666', 'o'),
    'native': ('Core native', '#2878b5', 's'),
    'native-cached': ('Core cached', '#d65b24', '^'),
    'warp-cached': ('Warp conditional', '#299464', 'D'),
}
DEGREE_COLORS = ['#386cb0', '#e58124', '#429a58', '#a75aa5']


def samples(row, *, isolated=False):
    if isolated:
        return row.get('isolated_matvec', row.get('isolated'))['graph_device_ms_per_call']
    return row['wall_ms']


def point(row, *, interactions=None, matrices=None, isolated=False, x=None):
    times = np.asarray(samples(row, isolated=isolated), dtype=float)
    if times.size < 2 or not np.isfinite(times).all() or not (times > 0).all():
        raise ValueError('paired plots require at least two finite, positive timing samples')
    median = float(np.median(times))
    if not isolated and not math.isclose(median, row['median_wall_ms'], rel_tol=1e-12):
        raise ValueError('stored median disagrees with raw samples')
    result = dict(x=row['n'] if x is None else x, n=row['n'], timing_samples_ms=times.tolist(),
                  median_ms=median, iqr_ms=np.quantile(times, [.25, .75]).tolist(),
                  interactions=interactions, matrices=matrices,
                  timing_scope='batch-graph device interval per action' if isolated else 'synchronized whole-call wall interval')
    for name, count in (('interaction', interactions), ('matrix', matrices)):
        if count is not None:
            if count < 0 or int(count) != count:
                raise ValueError('work counts must be nonnegative integers')
            rates = count * 1000 / times
            result[f'{name}_rate_per_second'] = float(np.median(rates))
            result[f'{name}_rate_iqr'] = np.quantile(rates, [.25, .75]).tolist()
    return result


def compact_n(n, _):
    return f'{n / 1e6:.3g}M' if n >= 1e6 else f'{n / 1e3:.3g}k'


def series(ax, points, *, metric, label, color, marker='o', linestyle='-', band=True):
    points = sorted(points, key=lambda p: p['x'])
    x = [p['x'] for p in points]
    if metric == 'time':
        y = [p['median_ms'] for p in points]
        bounds = np.asarray([p['iqr_ms'] for p in points])
    else:
        scale = 1e9 if metric == 'interaction' else 1e6
        y = [p[f'{metric}_rate_per_second'] / scale for p in points]
        bounds = np.asarray([p[f'{metric}_rate_iqr'] for p in points]) / scale
    ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.8, markersize=5, zorder=3)
    if band:
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=.13, linewidth=0)


def format_axis(ax, *, metric, particle_axis=True):
    if particle_axis:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(compact_n))
        ax.set_xlabel('Particles N')
    else:
        ax.set_xlabel('Actual neighbors per particle (including self)')
        ax.set_xticks([21, 57, 109, 221])
    if metric == 'time':
        ax.set_yscale('log')
        ax.set_ylabel('Time (ms)')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif metric == 'interaction':
        ax.set_ylabel('Directed non-self interactions/s (billions)')
        ax.set_ylim(bottom=0)
    elif metric == 'matrix':
        ax.set_ylabel('2×2 pseudoinverse instances/s (millions)')
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel('Executed matvec passes')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which='major', alpha=.22)
    ax.tick_params(labelsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scaling-input', type=Path, default=coreDevelopmentPath('docs/regression/cg_scaling_2026_10_15.json'))
    parser.add_argument('--accuracy-input', type=Path, default=coreDevelopmentPath('docs/regression/cg_scaling_accuracy_2026_10_15.json'))
    parser.add_argument('--operator-input', type=Path, default=coreDevelopmentPath('docs/regression/sph_work_2026_10_15.json'))
    parser.add_argument('--output-dir', type=Path, default=coreDevelopmentPath('docs/development/assets'))
    parser.add_argument('--manifest', type=Path, default=coreDevelopmentPath('docs/regression/scaling_figures_2026_10_15.json'))
    parser.add_argument('--prefix', default='scaling-2026-10-15')
    args = parser.parse_args()
    paths = [args.scaling_input, args.accuracy_input, args.operator_input]
    fixed, accuracy, operators = [json.loads(p.read_text()) for p in paths]
    if not all(report['complete'] for report in (fixed, accuracy, operators)):
        parser.error('all input benchmark reports must be complete')
    if fixed['configuration']['sph_rtol'] != 0 or accuracy['configuration']['sph_rtol'] <= 0:
        parser.error('fixed-work and equal-accuracy inputs must have their stated tolerance regimes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlepad': 10, 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})
    groups, accuracyGroups = defaultdict(list), defaultdict(list)
    gpu = fixed['cases'][0]['result']['gpu']
    if operators['dtype'] != 'torch.float32' or operators['gpu'] != gpu:
        parser.error('operator data must use float32 and the same GPU as the solver data')
    for data, target in ((fixed, groups), (accuracy, accuracyGroups)):
        for case in data['cases']:
            result = case['result']
            if result['dtype'] != 'torch.float32':
                parser.error('this baseline figure collection requires float32 throughout')
            row = result['records'][0]
            if result['gpu'] != gpu or len(set(row['matvecs'])) != 1:
                parser.error('paired curves require one GPU and constant per-sample work')
            degree = row['neighborhood']['mean'] if row['neighborhood'] else None
            target[(result['workload'], degree, result['engine'])].append(row)
    degrees = sorted({degree for workload, degree, _ in groups if workload == 'sph'})
    degreeStyles = dict(zip(degrees, DEGREE_COLORS))
    referenceDegree = min(degrees, key=lambda degree: abs(degree - 55))
    manifest = dict(inputs={str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve()):
                            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    statistic='median of raw samples; shaded 25th–75th percentiles, not confidence intervals',
                    throughput='same samples and interval as timing; actual directed non-self pairs × executed passes',
                    environment='RTX PRO 6000 Blackwell; float32; fixed 2D periodic lattices; setup excluded', figures=[])

    def save(fig, name, title, caption, data):
        fig.suptitle(title, fontsize=14)
        fig.text(.5, .915, f"{gpu.removeprefix('NVIDIA ').removesuffix(' Workstation Edition')} · float32 · periodic 2D lattice",
                 ha='center', fontsize=9, color='#555555')
        fig.text(.5, .015, caption, ha='center', va='bottom', fontsize=9, color='#444444')
        # Keep logarithmic decades but label the actual last sampled size, so
        # 4.2M/16.8M endpoints do not look like an unlabeled extension past 1M.
        for ax in fig.axes:
            # Setting a zero baseline while adding the first series freezes
            # its upper limit. Rescale after *all* series/bands are present.
            ax.set_autoscaley_on(True)
            ax.autoscale_view(scalex=False, scaley=True)
            if ax.get_yscale() == 'linear':
                ax.set_ylim(bottom=0)
            lowY, highY = ax.get_ylim()
            for line in ax.lines:
                if not all(lowY <= y <= highY for y in line.get_ydata()):
                    raise ValueError('figure would clip a plotted data point')
            if ax.get_xscale() != 'log':
                continue
            xs = [float(x) for line in ax.lines for x in line.get_xdata() if x > 0]
            low, high = min(xs), max(xs)
            ticks = [10. ** exponent for exponent in range(math.ceil(math.log10(low)), math.floor(math.log10(high)) + 1)]
            if ticks and ticks[0] / low < 2:
                ticks[0] = low
            else:
                ticks.insert(0, low)
            if high / ticks[-1] < 2:
                ticks[-1] = high
            else:
                ticks.append(high)
            ax.set_xticks(sorted(set(ticks)))
        fig.tight_layout(rect=(0, .09, 1, .94), h_pad=2.4, w_pad=2.)
        stem = args.output_dir / f'{args.prefix}-{name}'
        for extension in ('png', 'svg'):
            fig.savefig(stem.with_suffix(f'.{extension}'), dpi=180,
                        metadata={'Creator': 'warpSPHCore benchmark plots'} if extension == 'svg' else None)
        plt.close(fig)
        manifest['figures'].append(dict(name=name, title=title, caption=caption,
                                       png=str(stem.with_suffix('.png').resolve()),
                                       svg=str(stem.with_suffix('.svg').resolve()), series=data))
        print(stem.with_suffix('.png'))

    # Four stencil neighbors per row, excluding the central coefficient.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    data = {}
    for engine, (label, color, marker) in ENGINE_STYLE.items():
        rows = groups[('stencil', None, engine)]
        points = []
        for row in rows:
            if len(set(row['matvecs'])) != 1:
                raise ValueError('work varies across timing samples')
            points.append(point(row, interactions=4 * row['n'] * row['matvecs'][0]))
        data[label] = points
        for ax, metric in zip(axes, ('time', 'interaction')):
            series(ax, points, metric=metric, label=label, color=color, marker=marker)
            format_axis(ax, metric=metric)
    axes[0].set_title('Whole CG call: timing')
    axes[1].set_title('Same work and interval: throughput')
    axes[0].legend(fontsize=9)
    save(fig, 'stencil', 'Cheap stencil: scheduling floor and large-N scaling',
         '30 updates + 2 residual checks; four directed neighbor evaluations per row per matvec.\n'
         'Float32 · RTX PRO 6000 Blackwell · 7 samples · shaded IQR · fixed work, not equal accuracy.', data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    data = {}
    for engine, (label, color, marker) in ENGINE_STYLE.items():
        points = [point(row, interactions=row['neighborhood']['nonself_edges'] * row['matvecs'][0])
                  for row in groups[('sph', referenceDegree, engine)]]
        data[f'{label}, degree {referenceDegree:g}'] = points
        for ax, metric in zip(axes[0], ('time', 'interaction')):
            series(ax, points, metric=metric, label=label, color=color, marker=marker)
            format_axis(ax, metric=metric)
    for degree, color in degreeStyles.items():
        points = [point(row, interactions=row['neighborhood']['nonself_edges'] * row['matvecs'][0])
                  for row in groups[('sph', degree, 'native-cached')]]
        data[f'Core cached, degree {degree:g}'] = points
        for ax, metric in zip(axes[1], ('time', 'interaction')):
            series(ax, points, metric=metric, label=f'{degree:g} neighbors', color=color)
            format_axis(ax, metric=metric)
    axes[0, 0].set_title(f'Solver schedules · {referenceDegree:g} actual neighbors')
    axes[0, 1].set_title('Interaction throughput · same solver calls')
    axes[1, 0].set_title('Neighbor count · core cached CG')
    axes[1, 1].set_title('Interaction throughput · same degrees')
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].legend(fontsize=9)
    save(fig, 'cg', 'SPH CG: timing and interaction throughput',
         '8 scheduled updates + 2 residual checks; actual non-self CSR pairs × 10 matvecs.\n'
         'Synchronized wall intervals · float32 · 7 samples · shaded IQR · fixed work, not equal accuracy.', data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    data = {}
    for degree, color in degreeStyles.items():
        points = [point(row, interactions=row['neighborhood']['nonself_edges'], isolated=True)
                  for row in groups[('sph', degree, 'native-cached')]]
        data[f'{degree:g} neighbors'] = points
        for ax, metric in zip(axes, ('time', 'interaction')):
            series(ax, points, metric=metric, label=f'{degree:g} neighbors', color=color)
            format_axis(ax, metric=metric)
    for ax in axes:
        ax.axvspan(65536, 262144, color='#888888', alpha=.08, zorder=0)
    axes[0].set_title('Isolated SPH action: device interval')
    axes[1].set_title('Same action: interaction throughput')
    axes[0].legend(fontsize=9)
    save(fig, 'sph-action', 'SPH action: scaling and throughput saturation versus neighbor count',
         'Production Brookshaw Laplacian + identity scale/add; one pass, excluding self pairs.\n'
         'Warmed 16-call batch graph · 7 samples · shaded IQR; gray N band brackets the 2× small-N timing marker.', data)

    # Match target accuracy while exposing actual extra scheduled matvecs.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    data = {}
    selectedDegree = max(degrees)
    largestN = max(row['n'] for rows in accuracyGroups.values() for row in rows)
    for engine in ('native-cached', 'warp-cached'):
        label, color, marker = ENGINE_STYLE[engine]
        for rowIndex in (0, 1):
            rows = (accuracyGroups[('sph', selectedDegree, engine)] if rowIndex == 0 else
                    [row for degree in degrees for row in accuracyGroups[('sph', degree, engine)] if row['n'] == largestN])
            points = []
            for row in rows:
                if row['true_relative_residual'] > accuracy['configuration']['sph_rtol']:
                    raise ValueError('equal-accuracy figure contains an uncertified residual')
                work = row['neighborhood']['nonself_edges'] * row['matvecs'][0]
                if row['work']['evaluated_nonself_pairs'] != work:
                    raise ValueError('stored work ledger disagrees with plotted interactions')
                p = point(row, interactions=work, x=None if rowIndex == 0 else row['neighborhood']['mean'])
                p.update(matvecs=row['matvecs'][0], accepted_updates=row['completed_updates'],
                         scheduled_updates=row['scheduled_updates'], true_relative_residual=row['true_relative_residual'])
                points.append(p)
            data[f'{label}: ' + ('size sweep' if rowIndex == 0 else 'degree sweep')] = points
            for ax, metric in zip(axes[rowIndex], ('time', 'interaction', 'passes')):
                if metric == 'passes':
                    ordered = sorted(points, key=lambda p: p['x'])
                    ax.plot([p['x'] for p in ordered], [p['matvecs'] for p in ordered],
                            color=color, marker=marker, label=label)
                else:
                    series(ax, points, metric=metric, label=label, color=color, marker=marker)
                format_axis(ax, metric=metric, particle_axis=rowIndex == 0)
    for ax, title in zip(axes[0], (f'Timing · {selectedDegree:g} neighbors', 'Interaction throughput', 'Actual work, including checks')):
        ax.set_title(title)
    for ax, title in zip(axes[1], (f'Timing · N={largestN / 1e6:.2f}M', 'Interaction throughput', 'Actual work, including checks')):
        ax.set_title(title)
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].legend(fontsize=9)
    save(fig, 'accuracy', 'Equal accuracy: distinguish throughput from work needed to converge',
         'rtol=1e−5; all true residuals independently certified; shaded IQR over 7 samples.\n'
         'Whole-call wall timing includes frozen scheduled updates and checks. Warp conditional is a benchmark-only backend.', data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    data = {}
    names = ['density', 'interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance']
    palette = ['#2878b5', '#d65b24', '#299464', '#a45ca6', '#c44749', '#807130', '#537a83']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    for name, color, marker in zip(names, palette, markers):
        rows = [row for row in operators['records'] if row['operation'] == name]
        points = [point(row, interactions=row['edges'] - row['n'], isolated=True) for row in rows]
        for row, p in zip(rows, points):
            if not math.isclose(p['interaction_rate_per_second'], row['device_work']['directed_interactions_per_second'], rel_tol=1e-12):
                raise ValueError('operator throughput disagrees with measured work ledger')
        data[name] = points
        for ax, metric in zip(axes, ('time', 'interaction')):
            series(ax, points, metric=metric, label=name.capitalize(), color=color, marker=marker)
            format_axis(ax, metric=metric)
    axes[0].set_title('Public operator: device interval per call')
    axes[1].set_title('Same operator: interaction throughput')
    axes[0].legend(fontsize=9, ncol=2)
    save(fig, 'operators', 'Individual SPH operators: timing and pair throughput',
         '57 actual neighbors · directed non-self pairs · warmed 16-call graph · 7 samples · shaded IQR.\n'
         '2D/float32; scalar fields except vector divergence/curl; covariance has matrix output. Pair arithmetic differs by operator.', data)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    data = {}
    for name, label, color, marker in [('crk_factors', 'Complete CRK factors', '#2878b5', 'o'),
                                       ('crk_local_terms', 'Local coefficient stage', '#d65b24', 's')]:
        rows = [row for row in operators['records'] if row['operation'] == name]
        points = []
        for row in rows:
            count = 3 * (row['edges'] - row['n']) if name == 'crk_factors' else None
            if row['work']['evaluated_nonself_pairs'] != count:
                raise ValueError('CRK pair work disagrees with measured work ledger')
            pinv = [e for e in row['work']['entries'] if e['operation'] == 'CRK pseudoinverse']
            if len(pinv) != 1 or pinv[0]['matrices'] != row['n'] or (pinv[0]['rows'], pinv[0]['cols']) != (2, 2):
                raise ValueError('unexpected CRK local matrix work')
            points.append(point(row, interactions=count, matrices=pinv[0]['matrices']))
        data[label] = points
        series(axes[0], points, metric='time', label=label, color=color, marker=marker)
        if name == 'crk_factors':
            series(axes[1], points, metric='interaction', label='3 pair traversals / full-factor time', color=color, marker=marker)
        series(axes[2], points, metric='matrix', label=label, color=color, marker=marker)
    for ax, metric, title in zip(axes, ('time', 'interaction', 'matrix'),
                                 ('Whole-call timing', 'Full factors: pair throughput', 'Local-matrix work throughput')):
        format_axis(ax, metric=metric)
        ax.set_title(title)
    axes[0].legend(fontsize=9)
    axes[2].legend(fontsize=9)
    save(fig, 'crk', 'CRK factor construction: pair work and intra-particle matrix work',
         '57 neighbors · 3 pair traversals; N determinants and N 2×2 SVD pseudoinverses, then coefficient assembly.\n'
         'Matrix rates use the named stage wall time, not isolated SVD time. Stages timed independently; not an additive partition.', data)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Validated and saved {len(manifest["figures"])} paired figures; manifest: {args.manifest}')


if __name__ == '__main__':
    main()
