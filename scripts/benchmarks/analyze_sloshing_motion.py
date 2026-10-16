#!/usr/bin/env python3
"""Summarize matched sloshing windows and separately instrumented CSR rebuilds."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from development_paths import coreDevelopmentPath
import statistics


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def records(root):
    # A fresh reproduction's JSON supersedes an archived gzip of the same run.
    paths = {p.name.removesuffix('.gz'): p for p in root.glob('*-r*.json.gz')}
    paths.update({p.name: p for p in root.glob('*-r*.json')})
    return [read(p) | {'report_path': str(p)} for p in sorted(paths.values())]


def physical_rows(r):
    return [{k: v for k, v in row.items() if k != 'stepTime_ms'} for row in r['trajectory']]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=coreDevelopmentPath('docs/regression/sloshing_motion_2026_10_15'))
    ap.add_argument('--figure', type=Path, default=coreDevelopmentPath('docs/development/assets/sloshing-motion-2026-10-15.png'))
    args = ap.parse_args()
    trials, diagnostics, visual = [records(args.root / d) for d in ('trials', 'diagnostics', 'visual')]
    if (len(trials), len(diagnostics), len(visual)) != (24, 9, 2):
        raise RuntimeError(f'Expected 24 headline / 9 diagnostic / 2 visual runs, got {len(trials), len(diagnostics), len(visual)}')
    phases = ('startup', 'impact1', 'impact3')
    policies = ('wide-fused', 'narrow-fused', 'narrow-split', 'automatic')
    sources = None
    checks, aggregates, section_stats = [], [], []
    all_reports = trials + diagnostics + visual
    for phase in phases:
        group = [r for r in trials if r['fixture_phase'] == phase]
        reference = group[0]
        for r in [r for r in all_reports if r['fixture_phase'] == phase]:
            production = {p: h for p, h in r['source_sha256'].items() if '/src/' in p}
            if sources is None:
                sources = production
            if sources != production:
                raise RuntimeError('Production sources changed between trials')
            assert r['complete'] and r['final_state_finite'] and r['graph']['disabled'] is None
            checks.append(dict(report=r['report_path'], reference=reference['report_path'],
                               final_tensors_bitwise_equal=r['final_state'] == reference['final_state'],
                               physical_trajectory_equal=physical_rows(r) == physical_rows(reference),
                               csr_rebuild_history_equal=[(o['step'], o['edges']) for o in r['section_observations'] if o['section'] == 'search'] ==
                                                        [(o['step'], o['edges']) for o in reference['section_observations'] if o['section'] == 'search']))
        for policy in policies:
            group = [r for r in trials if r['fixture_phase'] == phase and r['policy'] == policy]
            timing = [r['mean_loop_ms'] for r in group]
            counts = [sum(o['section'] == 'search' and o['step'] >= r['warmup'] for o in r['section_observations']) for r in group]
            graph = [r['graph'] for r in group]
            aggregates.append(dict(phase=phase, policy=policy, particles=group[0]['particles'],
                                   trial_loop_ms=timing, median_trial_loop_ms=statistics.median(timing),
                                   active_steps=group[0]['active'], active_csr_build_calls=counts,
                                   graph_including_warmup=graph,
                                   peak_torch_allocated_bytes=[r['active_peak_torch_allocated_bytes'] for r in group],
                                   active_time_range=[group[0]['trajectory'][group[0]['warmup'] + 1]['t'], group[0]['trajectory'][-1]['t']]))
    for r in diagnostics:
        last_search = None
        paired = []
        for item in r['section_observations']:
            if item['section'] == 'search':
                last_search = item
            elif item['section'] == 'grid_and_search':
                if last_search is None or last_search['step'] != item['step']:
                    raise RuntimeError('Cannot associate complete rebuild with its CSR output')
                paired.append(item | {'edges': last_search['edges'], 'search_device_interval_ms': last_search.get('device_interval_ms')})
        active = [o for o in paired if o['step'] >= r['warmup'] and 'device_interval_ms' in o]
        total_ms = sum(o['device_interval_ms'] for o in active)
        total_pairs = sum(o['edges'] for o in active)
        section_stats.append(dict(phase=r['fixture_phase'], policy=r['policy'], observed_builds=len(active),
                                  mean_complete_rebuild_device_interval_ms=statistics.mean(o['device_interval_ms'] for o in active) if active else None,
                                  mean_prebuilt_grid_search_device_interval_ms=statistics.mean(o['search_device_interval_ms'] for o in active) if active else None,
                                  stored_csr_pairs_per_second=total_pairs / (total_ms / 1000) if total_ms else None,
                                  amortized_complete_rebuild_interval_ms_per_step=total_ms / r['active'],
                                  event_instrumented_loop_ms=r['mean_loop_ms'], samples=active))
    for path, digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Production source no longer matches: {path}')
    import numpy as np
    fixtures = read(args.root / 'fixtures/manifest.json')
    locality = {}
    initial = np.load(args.root / 'fixtures/startup.npz')
    for phase in phases:
        p = args.root / 'fixtures' / f'{phase}.npz'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == fixtures['fixtures'][phase]['sha256']
        a = np.load(p)
        mask = a['kinds'] == 0
        dx = .9 / 200
        displacement = np.linalg.norm(a['positions'][mask] - initial['positions'][mask], axis=1)
        adjacent = np.linalg.norm(np.diff(a['positions'][mask], axis=0), axis=1) / dx
        locality[phase] = dict(fluid_displacement_median_m=float(np.median(displacement)),
                               fluid_displacement_p95_m=float(np.quantile(displacement, .95)),
                               consecutive_fluid_row_distance_over_dx_p50=float(np.median(adjacent)),
                               consecutive_fluid_row_distance_over_dx_p95=float(np.quantile(adjacent, .95)))
    result = dict(complete=True, successful_trials=len(all_reports),
                  total_simulated_steps=sum(r['warmup'] + r['active'] for r in all_reports),
                  all_final_tensors_bitwise_equal=all(c['final_tensors_bitwise_equal'] for c in checks),
                  all_physical_trajectories_equal=all(c['physical_trajectory_equal'] for c in checks),
                  all_csr_rebuild_histories_equal=all(c['csr_rebuild_history_equal'] for c in checks),
                  source_sha256=sources, aggregates=aggregates, search_diagnostics=section_stats,
                  fixture_manifest_sha256=hashlib.sha256((args.root / 'fixtures/manifest.json').read_bytes()).hexdigest(),
                  fixture_locality=locality, comparisons=checks,
                  reports_sha256={r['report_path']: hashlib.sha256(Path(r['report_path']).read_bytes()).hexdigest() for r in all_reports},
                  timing_scope='Headline: two uninstrumented serial trials, 200 warmup + 1000 active steps, full pipelined loop with native case hooks/diagnostics. Diagnostic/visual trials excluded from headline means.',
                  work_scope='Throughput is stored directed Verlet CSR output pairs per complete grid+CSR rebuild interval. It is not physical active interactions/s for the whole SPH scheme. Event intervals include host-fed GPU idle gaps; diagnostic event overhead is included.',
                  fixture_scope='Integrated x/v/rho and static metadata from stored Oct 12 TC10 trajectories, current-runtime derived fields and fresh adjacency, roll/gravity set to frame time before first step. No particle reorder or claim of exact historical continuation.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#888888', '#e2a247', '#91b65b', '#267da5']
    labels = ['64-bit fused/input', '32-bit fused/input', '32-bit split/input', '32-bit automatic']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(3)
    for i, (policy, color, label) in enumerate(zip(policies, colors, labels)):
        rows = [next(r for r in aggregates if r['phase'] == p and r['policy'] == policy) for p in phases]
        values = [r['median_trial_loop_ms'] for r in rows]
        err = np.array([[v - min(r['trial_loop_ms']), max(r['trial_loop_ms']) - v] for v, r in zip(values, rows)]).T
        axes[0, 0].bar(x + (i - 1.5) * .19, values, width=.18, color=color, label=label,
                        yerr=err, error_kw={'elinewidth': 1, 'capsize': 2})
    counts = [next(r for r in aggregates if r['phase'] == p and r['policy'] == 'automatic')['active_csr_build_calls'][0] for p in phases]
    axes[0, 1].bar(x, counts, width=.5, color='#267da5')
    for i, value in enumerate(counts):
        axes[0, 1].text(i, value + 2, str(value), ha='center')
    for i, policy in enumerate(('wide-fused', 'narrow-fused', 'automatic')):
        color = colors[policies.index(policy)]
        rows = [next(r for r in section_stats if r['phase'] == p and r['policy'] == policy) for p in phases[1:]]
        axes[1, 0].bar(np.arange(2) + (i - 1) * .23, [r['mean_complete_rebuild_device_interval_ms'] for r in rows], width=.22, color=color)
        axes[1, 1].bar(np.arange(2) + (i - 1) * .23, [r['stored_csr_pairs_per_second'] / 1e9 for r in rows], width=.22, color=color)
    for ax in axes[0]:
        ax.set_xticks(x, ['Startup', 'First impact', 'Third impact'])
    for ax in axes[1]:
        ax.set_xticks([0, 1], ['First impact', 'Third impact'])
    axes[0, 0].set_ylabel('Full loop ms / step')
    axes[0, 1].set_ylabel('CSR build calls / 1,000 active steps')
    axes[1, 0].set_ylabel('Complete grid + CSR rebuild ms / call')
    axes[1, 1].set_ylabel('Stored CSR output pairs / second (billions)')
    axes[0, 0].set_title('Uninstrumented simulation timing')
    axes[0, 1].set_title('Natural rebuild activity (same for every policy)')
    axes[1, 0].set_title('Separate event-instrumented diagnostic pass')
    axes[1, 1].set_title('Rebuild throughput; no active startup rebuilds')
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.suptitle('Delta-SPH: startup versus developed sloshing', fontsize=15, y=.99)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(.5, .96), frameon=False, fontsize=9)
    fig.text(.5, .01, '10,660 particles (4,200 fluid) · fixed dt 1e-4 · RTX PRO 6000 Blackwell · float32 · default output pipeline\n'
             'Recorded flow fixtures; current simulation code. Bottom panels count stored Verlet pairs, not whole-scheme physical interactions.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .9))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    fig.savefig(args.figure.with_suffix('.svg'))
    plt.close(fig)
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['figures_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.figure, args.figure.with_suffix('.svg'))}
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('successful_trials', 'total_simulated_steps', 'all_final_tensors_bitwise_equal', 'all_physical_trajectories_equal', 'all_csr_rebuild_histories_equal')}, indent=2))


if __name__ == '__main__':
    main()
