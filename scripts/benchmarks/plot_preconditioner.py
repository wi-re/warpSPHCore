#!/usr/bin/env python3
"""Paired timing/throughput plots for local inverse actions and vector-SPH solves."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from development_paths import coreDevelopmentPath, artifactLabel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    datasets, inputs, outputs = [], {}, {}
    for precision in ('float32', 'float64'):
        path = coreDevelopmentPath(f'docs/regression/preconditioner_{precision}_2026_10_15.json')
        data = json.loads(path.read_text())
        for source, expected in data['sourceSha256'].items():
            assert hashlib.sha256((root/source).read_bytes()).hexdigest() == expected, source
        datasets.append(data)
        inputs[artifactLabel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for scope in ('application', 'solve'):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        for row, data in enumerate(datasets):
            if scope == 'application':
                for d, style in ((2, '-'), (3, '--')):
                    for kind, color, label in (('block_spd', '#2166ac', 'Torch'), ('block_native', '#b2182b', 'Warp')):
                        records = sorted([r for r in data['applications'] if r['d'] == d and r['kind'] == kind], key=lambda r:r['n'])
                        x = [r['n'] for r in records]
                        timing = [r['capturedApplication']['medianDeviceIntervalMs'] for r in records]
                        axes[row, 0].plot(x, [t*1e3 for t in timing], style, marker='o', color=color, label=f'{label}, {d}x{d}')
                        axes[row, 1].plot(x, [n/(t*1e-3)/1e9 for n,t in zip(x,timing)], style, marker='o', color=color, label=f'{label}, {d}x{d}')
                axes[row, 0].set_ylabel('Captured action (microseconds)')
                axes[row, 1].set_ylabel('Local block actions / s (billions)')
                axes[row, 0].set_title(data['dtype'], loc='left')
            else:
                for kind, color, label in (('none', '#777777', 'None'), ('diagonal', '#1b7837', 'Diagonal'),
                                           ('block_spd', '#2166ac', 'Block / Torch'), ('block_native', '#b2182b', 'Block / Warp')):
                    records = sorted([r for r in data['solves'] if r['kind'] == kind and r['meanDegree'] == 57], key=lambda r:r['n'])
                    x = [r['n'] for r in records]
                    axes[row, 0].plot(x, [r['timing']['medianWallMs'] for r in records], marker='o', color=color, label=label)
                    axes[row, 1].plot(x, [r['work']['directed_interactions_per_second']/1e9 for r in records], marker='o', color=color, label=label)
                axes[row, 0].set_ylabel('Whole verified solve (ms)')
                axes[row, 1].set_ylabel('Directed interactions / s (billions)')
                rtol = records[0]['rtol']
                axes[row, 0].set_title(f"{data['dtype']}, rtol={rtol:g}", loc='left')
            for axis in axes[row]:
                axis.set_xscale('log', base=2)
                axis.set_yscale('log')
                axis.set_xticks([1024, 65536, 1048576], ['1,024', '65,536', '1,048,576'])
                axis.set_xlabel('Local blocks' if scope == 'application' else 'Particles')
                axis.grid(True, which='both', alpha=.2)
        axes[0, 0].legend(fontsize=8)
        title = ('Fixed small-block inverse actions; warmed batch graph, setup excluded' if scope == 'application'
                 else 'Vector-SPH GMRES, 57 neighbors; complete solve, setup excluded\nActual pair work includes frozen steps and verification')
        fig.suptitle(title, fontsize=12)
        for suffix in ('.png', '.svg'):
            path = coreDevelopmentPath(f'docs/development/assets/preconditioner-{scope}-2026-10-15{suffix}')
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=170)
            outputs[artifactLabel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
    manifest = dict(inputs=inputs, outputs=outputs, plotSourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (coreDevelopmentPath('docs/regression/preconditioner_figures_2026_10_15.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    print('Saved paired timing/throughput application and solve figures.')


if __name__ == '__main__':
    main()
