"""Plot full-step scaling; keep failed stiff-EOS controls separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    pairs = {}
    for path in sorted(args.results.glob('*/result.json')):
        result = json.loads(path.read_text())
        config = result['arguments']
        if config['device'] != args.device or result['profiled'] or config.get('stage_profile', False) or config['snapshots']:
            continue
        if path.parent.name.startswith(('gpu-repeat-', 'gpu-repeat-large-', 'gpu-confirm-')):
            continue
        if args.device == 'cuda' and path.parent.name.startswith('gpu-small-cuda-'):
            continue
        case = config['case']
        if case == 'tgv-wc':
            if config.get('sound_speed') != 10. or config['order'] != 'morton':
                continue
        elif case == 'gresho':
            if config['scheme'] != 'CRKSPH':
                continue
        elif case == 'waveEquation':
            if config['dim'] != 3:
                continue
        else:
            continue
        key = (case, result['particles'], config['n_h'])
        pairs.setdefault(key, {})[config['operators']] = result
    rows = []
    for (case, particles, n_h), pair in sorted(pairs.items()):
        warp, native = pair['warp'], pair['native']
        rows.append(dict(case=case, particles=particles, n_h=n_h,
                         warp_ms=warp['median_step_ms'], native_ms=native['median_step_ms'],
                         speedup=warp['median_step_ms'] / native['median_step_ms'],
                         warp_mean_ms=float(np.mean(warp['step_times_ms'])),
                         native_mean_ms=float(np.mean(native['step_times_ms'])),
                         candidates=native.get('csr_candidates_per_particle', native.get('neighbors')),
                         warmup=native['arguments']['warmup'], steps=native['arguments']['steps'],
                         warp_intervals_ms=warp['median_runner_interval_ms'],
                         native_intervals_ms=native['median_runner_interval_ms']))
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = 'cpu' if args.device == 'cpu' else 'gpu'
    (args.out / f'{prefix}-summary.json').write_text(json.dumps(rows, indent=2))

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), sharey=True)
    for axis, case, title in zip(axes, ('tgv-wc', 'gresho', 'waveEquation'),
                                 ('Delta-SPH · 2D TGV', 'CRKSPH · 2D Gresho', 'Wave equation · 3D')):
        samples = sorted((key, value) for key, value in pairs.items() if key[0] == case and key[2] == 4.)
        for backend, color, label in [('warp', '#3165ad', 'Warp core'), ('native', '#cf5728', 'Native core')]:
            x, median, lower, upper = [], [], [], []
            for key, pair in samples:
                times = np.asarray(pair[backend]['step_times_ms'])
                value = float(np.median(times))
                x.append(key[1]); median.append(value)
                lower.append(value - np.percentile(times, 25))
                upper.append(np.percentile(times, 75) - value)
            axis.errorbar(x, median, yerr=[lower, upper], marker='o', capsize=3,
                          color=color, label=label, linewidth=1.7)
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set_title(title, fontweight='bold')
        axis.set_xlabel('Particles')
        counts = [key[1] for key, _ in samples]
        axis.set_xticks(counts, [f'{value:,}' for value in counts])
        axis.tick_params(axis='x', labelsize=9)
        axis.grid(alpha=.2, which='both')
    axes[0].set_ylabel('Median complete step [ms] · lower is faster')
    axes[0].legend(frameon=False)
    hardware = 'CPU · 24 Torch threads' if args.device == 'cpu' else 'GPU · RTX PRO 6000 Blackwell'
    fig.suptitle(f'Full simulation steps on {hardware} · FP32 · nₕ = 4', fontsize=14, fontweight='bold')
    fig.text(.5, .02, 'Fluids retain Warp frontend kernels. Morton-ordered fluids; fixed wave geometry. Error bars: step IQR.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(args.out / f'{prefix}-scaling.png', dpi=180)
    fig.savefig(args.out / f'{prefix}-scaling.svg')


if __name__ == '__main__':
    main()
