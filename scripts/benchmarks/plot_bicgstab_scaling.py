#!/usr/bin/env python3
"""Validate and plot dense BiCGStab sweeps, with paired time/work-rate panels.

Preserves the original three-size qualification figures. Accepts any matched
particle-count grid; shaded ranges are the observed repeat min/max, not CIs.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


STYLES = {
    'torch': ('#667085', 'Torch'),
    'native': ('#2166ac', 'Native eager'),
    'native-cached': ('#b2182b', 'Native cached'),
    'native-out': ('#4393c3', 'Native eager / out'),
    'native-cached-out': ('#ef8a62', 'Native cached / out'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def particle_label(value, _position=None):
    if value >= 1e6:
        return f'{value/1e6:.2f}M'
    return f'{value/1e3:.3g}k'


def validate(data, root):
    for source, expected in data['sourceSha256'].items():
        if digest(root/source) != expected:
            raise ValueError(f'Benchmark source changed: {source}')
    keys = set()
    grids = {}
    for row in data['records']:
        key = (row['mode'], row['meanDegree'], row['engine'], row['n'])
        if key in keys:
            raise ValueError(f'Duplicate configuration: {key}')
        keys.add(key)
        grids.setdefault(key[:3], []).append(row['n'])
        samples = row['wallMs']
        assert len(samples) == data['arguments']['repeat'], key
        if not all(math.isfinite(t) and t > 0 for t in samples):
            raise ValueError(f'Invalid timing samples: {key}')
        assert row['medianWallMs'] == statistics.median(samples), key
        work = row['work']['evaluated_nonself_pairs']
        assert work == (row['edges']-row['n'])*row['matvecs'], key
        assert math.isclose(row['work']['directed_interactions_per_second'],
                            work/(row['medianWallMs']/1000), rel_tol=1e-12), key
        assert row['setupIterations'] == 0, key
        if row['mode'] == 'fixed':
            assert row['loops'] == data['arguments']['fixed_steps'], key
        else:
            assert row['status'] >= 0 and row['relativeResidual'] <= row['rtol'], key
    reference = sorted(next(iter(grids.values())))
    assert reference == sorted(side**2 for side in data['arguments']['sides']), 'Missing size'
    assert all(sorted(grid) == reference for grid in grids.values()), 'Unmatched size grids'
    assert {key[0] for key in keys} == {'fixed', 'verified'}, 'Missing mode'
    assert {key[2] for key in keys} == set(data['arguments']['engines']), 'Missing engine'
    assert {row['requestedNeighbors'] for row in data['records']} == set(data['arguments']['neighbors']), 'Missing neighborhood'
    assert len(grids) == 2*len(data['arguments']['neighbors'])*len(data['arguments']['engines']), 'Missing series'


def series_summary(rows):
    """Keep observed local slopes; do not force a single overhead+linear fit."""
    slopes = []
    for before, after in zip(rows, rows[1:]):
        log_size = math.log(after['n']/before['n'])
        slopes.append(dict(
            fromN=before['n'], toN=after['n'],
            wallExponent=math.log(after['medianWallMs']/before['medianWallMs'])/log_size,
            workExponent=math.log(after['work']['evaluated_nonself_pairs']/
                                  before['work']['evaluated_nonself_pairs'])/log_size,
        ))
    rates = [row['work']['directed_interactions_per_second'] for row in rows]
    return dict(
        n=[row['n'] for row in rows],
        medianWallMs=[row['medianWallMs'] for row in rows],
        minWallMs=[min(row['wallMs']) for row in rows],
        maxWallMs=[max(row['wallMs']) for row in rows],
        directedInteractionsPerSecond=rates,
        iterations=[row['iterations'] for row in rows],
        scheduledIterations=[row['loops'] for row in rows],
        matvecs=[row['matvecs'] for row in rows],
        peakRateN=rows[rates.index(max(rates))]['n'],
        peakDirectedInteractionsPerSecond=max(rates),
        adjacentSlopes=slopes,
    )


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, default=root/'docs/development/assets')
    parser.add_argument('--stem', default='bicgstab-scaling-2026-10-15')
    parser.add_argument('--manifest', type=Path,
                        default=root/'docs/regression/bicgstab_scaling_figures_2026_10_15.json')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(inputs={}, outputs={}, series=[],
                    plotSourceSha256=digest(Path(__file__)),
                    rangeMeaning='observed repeat min/max, not confidence intervals',
                    slopeMeaning='log(time ratio)/log(N ratio); near zero is flat, near one is linear; inspect workExponent too')
    for path in args.inputs:
        data = json.loads(path.read_text())
        validate(data, root)
        manifest['inputs'][str(path.resolve().relative_to(root))] = digest(path)
        precision = data['dtype'].removeprefix('torch.')
        degrees = sorted({row['meanDegree'] for row in data['records']})
        for mode in ('fixed', 'verified'):
            fig, axes = plt.subplots(len(degrees), 2, figsize=(12, 3.8*len(degrees)),
                                     squeeze=False, constrained_layout=True)
            for row_index, degree in enumerate(degrees):
                selection = [row for row in data['records']
                             if row['mode'] == mode and row['meanDegree'] == degree]
                all_sizes = sorted({row['n'] for row in selection})
                for engine, (color, label) in STYLES.items():
                    rows = sorted([row for row in selection if row['engine'] == engine],
                                  key=lambda row: row['n'])
                    if not rows:
                        continue
                    summary = series_summary(rows)
                    manifest['series'].append(dict(dtype=data['dtype'], mode=mode,
                                                   degree=degree, engine=engine, **summary))
                    x = summary['n']
                    low, high = summary['minWallMs'], summary['maxWallMs']
                    pairs = [row['work']['evaluated_nonself_pairs'] for row in rows]
                    rates = [rate/1e9 for rate in summary['directedInteractionsPerSecond']]
                    axes[row_index, 0].plot(x, summary['medianWallMs'], 'o-',
                                            color=color, label=label, markersize=4)
                    axes[row_index, 0].fill_between(x, low, high, color=color, alpha=.12)
                    axes[row_index, 1].plot(x, rates, 'o-', color=color, label=label, markersize=4)
                    axes[row_index, 1].fill_between(x, [p/t/1e6 for p, t in zip(pairs, high)],
                                                   [p/t/1e6 for p, t in zip(pairs, low)],
                                                   color=color, alpha=.12)
                # A slope guide, not a fitted or claimed hardware bound.
                cached = sorted([row for row in selection if row['engine'] == 'native-cached'],
                                key=lambda row: row['n'])
                if cached and mode == 'fixed':
                    tail = cached[-1]
                    axes[row_index, 0].plot(all_sizes, [tail['medianWallMs']*n/tail['n'] for n in all_sizes],
                                            ':', color='#333333', alpha=.5, label='Linear N guide')
                    axes[row_index, 1].axhline(tail['work']['directed_interactions_per_second']/1e9,
                                               ls=':', color='#333333', alpha=.5, label='Constant rate guide')
                if mode == 'verified':
                    for before, after in zip(cached, cached[1:]):
                        if before['loops'] != after['loops']:
                            axes[row_index, 0].annotate(
                                f"{before['loops']} → {after['loops']} scheduled steps",
                                xy=(after['n'], after['medianWallMs']), xytext=(-105, 18),
                                textcoords='offset points', fontsize=8,
                                arrowprops=dict(arrowstyle='->', color='#444444'),
                                bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
                for axis in axes[row_index]:
                    axis.set_xscale('log', base=2)
                    axis.set_yscale('log')
                    ticks = all_sizes[::2]
                    if all_sizes[-1] not in ticks:
                        if all_sizes[-1]/ticks[-1] < 2.5:
                            ticks[-1] = all_sizes[-1]
                        else:
                            ticks.append(all_sizes[-1])
                    axis.set_xticks(ticks)
                    axis.xaxis.set_major_formatter(FuncFormatter(particle_label))
                    axis.set_xlabel('Particles (every marker is a measured size)')
                    axis.grid(True, which='both', alpha=.2)
                # Avoid extending the y range down to the linear guide's tiny-N extrapolation.
                times = [t for r in selection for t in r['wallMs']]
                axes[row_index, 0].set_ylim(min(times)*.7, max(times)*1.4)
                axes[row_index, 0].set_ylabel('Whole call (ms)')
                axes[row_index, 1].set_ylabel('Directed interactions / s (billions)')
                axes[row_index, 0].set_title(f'{degree:g} CSR neighbors per particle (includes self)', loc='left')
                axes[row_index, 1].set_title(f'{degree-1:g} nonself interactions per matvec per particle', loc='left')
            for axis in axes[0]:
                axis.legend(fontsize=8, loc='upper left')
            target = (f"{data['arguments']['fixed_steps']} scheduled steps" if mode == 'fixed'
                      else f"verified solves, rtol={selection[0]['rtol']:g}")
            fig.suptitle(f'BiCGStab scaling — {precision}, {target}\n'
                         'Warmed whole calls; median and observed min–max; includes verification and frozen callbacks',
                         fontsize=12)
            for suffix in ('.png', '.svg'):
                output = args.output_dir/f'{args.stem}-{precision}-{mode}{suffix}'
                fig.savefig(output, dpi=170)
                manifest['outputs'][str(output.resolve().relative_to(root))] = digest(output)
            plt.close(fig)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"Validated {sum(len(s['n']) for s in manifest['series'])} configurations; saved {len(manifest['outputs'])} figures.")


if __name__ == '__main__':
    main()
