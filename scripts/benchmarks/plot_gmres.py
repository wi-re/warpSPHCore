#!/usr/bin/env python3
"""Plot matched GMRES time and observed interaction rate from saved baselines."""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    sources = []
    labels = [('float32', 'FP32 vectors / FP32 accumulation, rtol = 2e-5'),
              ('mixed', 'FP32 vectors / FP64 accumulation, rtol = 2e-5'),
              ('float64', 'FP64 vectors / FP64 accumulation, rtol = 1e-10')]
    for row, (name, title) in enumerate(labels):
        source = coreDevelopmentPath(f'docs/regression/gmres_sph_{name}_2026_10_15.json')
        data = json.loads(source.read_text())
        for path, expected in data['sourceSha256'].items():
            if hashlib.sha256((args.root / path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'source hash changed: {path}')
        sources.append(source)
        for degree, color in [(21, '#2166ac'), (57, '#b2182b')]:
            for side, style in [('right', '-'), ('left', '--')]:
                records = sorted([r for r in data['records'] if r['meanDegree'] == degree
                                  and r['preconditionerSide'] == side], key=lambda r: r['n'])
                x = [r['n'] for r in records]
                label = f'{degree} neighbors, {side}'
                axes[row, 0].plot(x, [r['medianWallMs'] for r in records], style, marker='o', color=color, label=label)
                axes[row, 1].plot(x, [r['work']['directed_interactions_per_second'] / 1e9 for r in records],
                                  style, marker='o', color=color, label=label)
        axes[row, 0].set_title(title, fontsize=10, loc='left')
        axes[row, 0].set_ylabel('Whole verified solve (ms)')
        axes[row, 1].set_ylabel('Directed interactions / s (billions)')
        for axis in axes[row]:
            axis.set_xscale('log', base=2)
            axis.set_yscale('log')
            axis.set_xticks([1024, 65536, 1048576], ['1,024', '65,536', '1,048,576'])
            axis.grid(True, which='both', alpha=.2)
            axis.set_xlabel('Particles')
    axes[0, 1].legend(fontsize=8, loc='upper left')
    fig.suptitle('Restarted GMRES on a fixed nonsymmetric SPH operator\n'
                 'restart 20; observed work includes frozen callbacks and residual verification', fontsize=12)
    destination = coreDevelopmentPath('docs/development/assets/gmres-scaling-2026-10-15')
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ('.png', '.svg'):
        path = destination.with_suffix(suffix)
        fig.savefig(path, dpi=170)
        outputs.append(path)
    plt.close(fig)
    manifest = dict(inputs={artifactLabel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    outputs={artifactLabel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
                    plotSourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (coreDevelopmentPath('docs/regression/gmres_figures_2026_10_15.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(destination.with_suffix('.png'))


if __name__ == '__main__':
    main()
