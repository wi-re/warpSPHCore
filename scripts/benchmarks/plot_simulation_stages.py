"""Plot exclusive frontend stage costs, with hatching for remaining Warp work.

Input is a campaign directory with timing-{cpu,cuda}-{scheme}-{backend}-stages
and matching -control directories. Colors encode simulation aspects; texture
encodes actual dispatch. Raw individual frontend gates remain in the JSON.
"""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ASPECTS = {
    'Neighborhood / support': '#4477aa',
    'CRK factors': '#aa66aa',
    'Differential operators': '#228833',
    'Diffusion / switches': '#66ccee',
    'Forces / pair work': '#ee7733',
    'EOS / other physics': '#ccbb44',
    'Layout / integration': '#bbbbbb',
}
ENGINES = {'native': '', 'Warp': '///', 'Torch/Python': '...'}


def aspect(stage):
    name = stage.lower()
    if any(s in name for s in ('adjacency', 'optimal support', 'support clamp')):
        return 'Neighborhood / support'
    if 'crk factors' in name:
        return 'CRK factors'
    if any(s in name for s in ('pressure force', 'pressure / viscosity', 'wall pressure', 'dudt',
                                'de/dt', 'energy-partition', 'pair work', 'compatible energy')):
        return 'Forces / pair work'
    if any(s in name for s in ('diffusion', 'viscosity-switch')):
        return 'Diffusion / switches'
    if any(s in name for s in ('gradient', 'gradrho', 'continuity', 'divergence', 'laplacian', 'drho/dt')):
        return 'Differential operators'
    if name == 'layout / integration' or name.endswith(('] update', '] enforce updates')):
        return 'Layout / integration'
    return 'EOS / other physics'


def load(root, device, scheme, backend):
    base = root / f'timing-{device}-{scheme}-{backend}'
    stages = json.loads(Path(str(base) + '-stages/stage-timing.json').read_text())
    result = json.loads(Path(str(base) + '-stages/result.json').read_text())
    control = json.loads(Path(str(base) + '-control/result.json').read_text())
    costs = defaultdict(float)
    for item in stages['mean_costs']:
        costs[(aspect(item['stage']), item['engine'])] += item['ms']
    return stages, result, control, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'hatch.linewidth': .7,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    summaries = []
    for row, device in enumerate(('cpu', 'cuda')):
        for col, (scheme, title) in enumerate((('deltaSPH', 'Delta-SPH · TGV · 2D'),
                                              ('CRKSPH', 'CRKSPH · Gresho · 2D'),
                                              ('waveEquation', 'Wave equation · 3D'))):
            ax = axes[row, col]
            max_total = 0
            for x, backend in enumerate(('warp', 'native')):
                stages, result, control, costs = load(args.results, device, scheme, backend)
                total = 0
                for group, color in ASPECTS.items():
                    for engine, hatch in ENGINES.items():
                        value = costs.get((group, engine), 0)
                        if value:
                            ax.bar(x, value, bottom=total, width=.57, color=color,
                                   hatch=hatch, edgecolor='#333333', linewidth=.35)
                            total += value
                plain = control['mean_step_ms']
                ax.plot(x, plain, 'D', color='black', markersize=5, zorder=5)
                ax.annotate(f'{total:.2f} ms', (x, total), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=10)
                max_total = max(max_total, total, plain)
                engine_costs = {e: sum(v for (g, ee), v in costs.items() if ee == e) for e in ENGINES}
                summaries.append(dict(device=device, scheme=scheme, backend=backend, particles=result['particles'],
                                      profiled_mean_ms=total, plain_mean_ms=plain,
                                      perturbation_percent=100*(total/plain-1),
                                      engine_ms=engine_costs,
                                      aspect_ms={g: sum(v for (gg,e),v in costs.items() if gg==g) for g in ASPECTS},
                                      partition_timer_difference_ms=stages['mean_timer_ms']-total,
                                      detailed_costs=stages['mean_costs']))
            ax.set_ylim(0, max_total * 1.18)
            ax.set_xlim(-.6, 1.6)
            ax.set_xticks([0, 1], ['Warp core', 'Native core' if scheme == 'waveEquation' else 'Native core\n(mixed frontend)'])
            hardware = 'CPU · 24 Torch threads' if device == 'cpu' else 'GPU · RTX PRO 6000'
            ax.set_title(f'{title}\n{hardware} · N = {result["particles"]:,}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean elapsed step [ms]')
            ax.yaxis.grid(alpha=.2)
            ax.set_axisbelow(True)
    fig.suptitle('Timestep cost by simulation stage · FP32 · nₕ = 4', fontsize=17, fontweight='bold', y=.98)
    colors = [Patch(facecolor=c, label=g) for g,c in ASPECTS.items()]
    textures = [Patch(facecolor='white', edgecolor='#333333', hatch=h, label=label)
                for e,h,label in [('native','','Native C++ / CUDA'), ('Warp','///','Remaining Warp'),
                                  ('Torch/Python','...','Scaffolding / Torch / dispatch')]]
    textures.append(Line2D([0],[0],marker='D',color='black',linestyle='none',label='Ordinary run (no stage instrumentation)'))
    fig.legend(handles=colors, loc='lower center', bbox_to_anchor=(.5,.082), ncol=4, frameon=False, fontsize=9)
    fig.legend(handles=textures, loc='lower center', bbox_to_anchor=(.5,.035), ncol=4, frameon=False, fontsize=9)
    fig.text(.5,.009,'Independent linear scales. Integrator + layout only; excludes setup, diagnostics and output.\nElapsed CUDA time includes launch gaps and profiling overhead. Fluids: Morton order; wave: fixed geometry.',
             ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.17,1,.955),h_pad=2.5)
    for ext in ('png','svg'):
        fig.savefig(args.out/f'simulation-stages.{ext}',dpi=180)
    (args.out/'stage-summary.json').write_text(json.dumps(summaries,indent=2))
    with (args.out/'frontend-gates.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=['device','scheme','backend','particles','stage','engine','ms'])
        writer.writeheader()
        for row in summaries:
            for cost in row['detailed_costs']:
                writer.writerow({**{key: row[key] for key in ('device','scheme','backend','particles')}, **cost})


if __name__ == '__main__':
    main()
