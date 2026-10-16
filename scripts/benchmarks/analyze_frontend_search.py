#!/usr/bin/env python3
"""Check frontend trial records and plot full-loop time and Torch peak memory."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def trajectory(report):
    return [{k: v for k, v in row.items() if k != 'stepTime_ms'} for row in report['trajectory']]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directories', nargs='+', type=Path, help='primary sweep first, supplementary sweeps after')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--figure', type=Path, required=True)
    args = ap.parse_args()
    all_reports, summaries, checks, production_sources = [], [], [], None
    for directory in args.directories:
        reports = [json.loads(p.read_text()) | {'report_path': str(p)} for p in sorted(directory.glob('*-r*.json'))]
        if not reports:
            raise RuntimeError(f'no trials in {directory}')
        scenarios = list(dict.fromkeys(r['scenario'] for r in reports))
        aggregates = []
        for scenario in scenarios:
            group = [r for r in reports if r['scenario'] == scenario]
            reference = group[0]
            for r in group:
                if not r['complete'] or not r['final_state_finite']:
                    raise RuntimeError(f'incomplete trial {r["report_path"]}')
                sources = {p: h for p, h in r['source_sha256'].items() if '/src/' in p}
                if production_sources is None:
                    production_sources = sources
                if sources != production_sources:
                    raise RuntimeError('production sources differ between trials')
                keys = ('step', 'particles', 'edges', 'capacity')
                checks.append(dict(report=r['report_path'], reference=reference['report_path'],
                                   final_tensor_hashes_equal=r['final_state'] == reference['final_state'],
                                   trajectory_scalars_equal=trajectory(r) == trajectory(reference),
                                   search_rebuild_history_equal=[{k: c[k] for k in keys} for c in r['search_dispatches']] ==
                                                               [{k: c[k] for k in keys} for c in reference['search_dispatches']]))
            for policy in dict.fromkeys(r['policy'] for r in group):
                trials = [r for r in group if r['policy'] == policy]
                loop = [r['mean_loop_ms'] for r in trials]
                integrator = [r['mean_integrator_ms'] for r in trials]
                active_builds = [len([c for c in r['search_dispatches'] if c['step'] >= r['warmup']]) for r in trials]
                aggregates.append(dict(scenario=scenario, policy=policy, particles=trials[0]['particles'], trials=len(trials),
                                       pipeline_outputs=trials[0]['compute']['pipelineOutputs'],
                                       loop_ms=loop, median_trial_loop_ms=statistics.median(loop),
                                       integrator_ms=integrator, median_trial_integrator_ms=statistics.median(integrator),
                                       peak_torch_allocated_bytes=[r['active_peak_torch_allocated_bytes'] for r in trials],
                                       active_python_csr_builds=active_builds,
                                       graph=[r['graph'] for r in trials],
                                       resolved_active_search_policies=sorted({(c['collection'], c['order']) for r in trials
                                                                             for c in r['search_dispatches'] if c['step'] >= r['warmup']})))
        local_checks = [c for c in checks if c['report'] in {r['report_path'] for r in reports}]
        summaries.append(dict(directory=str(directory), aggregates=aggregates,
                              all_final_tensors_bitwise_equal=all(c['final_tensor_hashes_equal'] for c in local_checks),
                              all_numeric_trajectories_equal=all(c['trajectory_scalars_equal'] for c in local_checks),
                              all_search_rebuild_histories_equal=all(c['search_rebuild_history_equal'] for c in local_checks)))
        all_reports.extend(reports)
    mismatches = [p for p, h in production_sources.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    if mismatches:
        raise RuntimeError(f'production sources changed: {mismatches}')
    # Different grid construction can change the order of neighbor sums. Give
    # quantitative errors for that diagnostic override, rather than silently
    # treating a hash mismatch as either a failure or an exact match.
    import torch
    by_path = {r['report_path']: r for r in all_reports}
    for check in checks:
        if check['final_tensor_hashes_equal']:
            continue
        a_report, b_report = by_path[check['reference']], by_path[check['report']]
        a_state = torch.load(a_report['final_state_file'], weights_only=True)
        b_state = torch.load(b_report['final_state_file'], weights_only=True)
        differences = {}
        for key, a in a_state.items():
            b = b_state[key]
            max_abs = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
            scale = max(float(a.double().abs().max()), float(b.double().abs().max()), 1.) if a.numel() else 1.
            differences[key] = dict(max_abs_difference=max_abs, max_abs_over_max_one_field_norm=max_abs / scale,
                                    integer_exact=bool(torch.equal(a, b)) if not a.is_floating_point() else None)
        check['final_tensor_differences'] = differences
        diagnostics = {}
        for a, b in zip(trajectory(a_report), trajectory(b_report)):
            for key in a.keys() & b.keys():
                if a[key] is not None and b[key] is not None:
                    diagnostics[key] = max(diagnostics.get(key, 0.), abs(a[key] - b[key]))
        check['trajectory_max_abs_differences'] = diagnostics
    frontend_root = next(Path(p).parents[2] for p in production_sources if p.endswith('/src/warpSPH/__init__.py'))
    historical_paths = [frontend_root / p for p in ('docs/compsph_step/README.md', 'SCHEME_CLEANUP_PLAN.md', 'docs/microbench/README.md')]
    primary_by_key = {(r['scenario'], r['policy']): r for r in all_reports
                      if Path(r['report_path']).parent == args.directories[0]}
    pipeline_checks = []
    for r in all_reports:
        if not r['compute']['pipelineOutputs']:
            continue
        baseline = primary_by_key[(r['scenario'], r['policy'])]
        pipeline_checks.append(dict(report=r['report_path'], reference=baseline['report_path'],
                                    final_tensor_hashes_equal=r['final_state'] == baseline['final_state'],
                                    trajectory_scalars_equal=trajectory(r) == trajectory(baseline)))
    result = dict(complete=True, successful_trials=len(all_reports),
                  total_simulated_steps=sum(r['warmup'] + r['active'] for r in all_reports),
                  all_final_tensors_bitwise_equal=all(c['final_tensor_hashes_equal'] for c in checks),
                  all_numeric_trajectories_equal=all(c['trajectory_scalars_equal'] for c in checks),
                  all_search_rebuild_histories_equal=all(c['search_rebuild_history_equal'] for c in checks),
                  production_sources_match_current=True, source_sha256=production_sources,
                  benchmark_revisions=sorted({r['source_sha256'][str(Path(__file__).with_name('bench_frontend_search.py'))] for r in all_reports}),
                  primary_final_tensors_bitwise_equal=summaries[0]['all_final_tensors_bitwise_equal'],
                  primary_numeric_trajectories_equal=summaries[0]['all_numeric_trajectories_equal'],
                  historical_sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in historical_paths},
                  pipeline_vs_sequential=pipeline_checks,
                  sweeps=summaries, comparisons=checks,
                  reports_sha256={r['report_path']: hashlib.sha256(Path(r['report_path']).read_bytes()).hexdigest() for r in all_reports},
                  timing_scope='Trial means over steps 40–199 (zero-based), then median of two trials. Primary/attribution/grid sweeps: runner CUDA-event integrator interval includes host-fed GPU idle gaps; completed-diagnostics wall intervals include timestep/diagnostics/bookkeeping; pipeline off. Pipeline supplement: completed-diagnostics intervals measure overlapped throughput, and runner stepTime_ms is wall throughput too. No profiler/video. Torch memory excludes non-Torch allocations.',
                  baseline_scope='Current production sources throughout. wide-fused: 64-bit IDs/offsets + fused/input. narrow-fused: 32-bit IDs/offsets + fused/input. automatic: shipped 32-bit defaults. These are controlled policy comparisons, not historical checkouts.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    primary = summaries[0]['aggregates']
    names = list(dict.fromkeys(r['scenario'] for r in primary))
    labels = [f'{s}\nN = {next(r["particles"] for r in primary if r["scenario"] == s):,}' for s in names]
    policies = [('wide-fused', '64-bit, fused / input', '#8a8a8a'),
                ('narrow-fused', '32-bit, fused / input', '#e2a247'),
                ('automatic', '32-bit, automatic', '#267da5')]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9.5), sharey=True, gridspec_kw={'width_ratios': [1.4, 1]})
    y = np.arange(len(names))
    for i, (policy, label, color) in enumerate(policies):
        rows = [next(r for r in primary if r['scenario'] == s and r['policy'] == policy) for s in names]
        values = [r['median_trial_loop_ms'] for r in rows]
        errors = np.array([[v - min(r['loop_ms']), max(r['loop_ms']) - v] for r, v in zip(rows, values)]).T
        axes[0].errorbar(values, y + (i - 1) * .23, xerr=errors, fmt='o', color=color,
                         label=label, markersize=5.5, elinewidth=1, capsize=2)
        memory = [statistics.median(r['peak_torch_allocated_bytes']) / 1e9 for r in rows]
        axes[1].barh(y + (i - 1) * .23, memory, height=.21, color=color)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xscale('log')
    axes[0].set_xticks([2, 3, 5, 10, 20, 40, 80], labels=['2', '3', '5', '10', '20', '40', '80'])
    axes[0].minorticks_off()
    axes[0].set_xlabel('Full loop ms / step (log scale)')
    axes[1].set_xlabel('Active peak Torch allocation (GB)')
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frontend simulations: search / indexing policy comparison', fontsize=15, y=.985)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(.57, .955), frameon=False)
    fig.text(.5, .015, 'RTX PRO 6000 Blackwell · float32 · 40 warmup + 160 active steps · two serial trials · output pipeline off\n'
             'Same current sources. Timing points / memory bars compare policies; error bars span trial means. No profiler or video.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .92))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    fig.savefig(args.figure.with_suffix('.svg'))
    plt.close(fig)
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['figures_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.figure, args.figure.with_suffix('.svg'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('successful_trials', 'total_simulated_steps', 'all_final_tensors_bitwise_equal',
                                            'all_numeric_trajectories_equal', 'all_search_rebuild_histories_equal')}, indent=2))


if __name__ == '__main__':
    main()
