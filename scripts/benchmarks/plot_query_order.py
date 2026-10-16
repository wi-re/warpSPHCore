#!/usr/bin/env python3
"""Paired timing/rate plots for mapped-query scheduling and collection choices."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_neighbor_search import draw, finishAxes, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.inputs]
    if not all(r['complete'] and r['configuration']['device'] == 'cuda' for r in reports): parser.error('completed CUDA reports required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    manifest = dict(complete=False, figures=[], cases=[],
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
        helper_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('plot_query_order.py', 'plot_neighbor_search.py', 'plot_work_scaling.py')})
    styles = [('input_fused', 'Input + fused', '#666666', 'o'), ('input_split', 'Input + split', '#2878b5', 's'),
              ('spatial_fused', 'Spatial + fused', '#d65b24', '^'), ('spatial_split', 'Spatial + split', '#299464', 'D')]
    for report in reports:
        dim = report['configuration']['dim']; degrees = report['configuration']['neighbors']
        for scope in ('public', 'collect', 'count'):
            fig, axes = plt.subplots(len(degrees), 2, figsize=(12, 4 * len(degrees)), squeeze=False)
            for index, degree in enumerate(degrees):
                for layout in report['configuration']['layouts']:
                    rows = sorted((r for r in report['records'] if r['requested_neighbors'] == degree and r['layout'] == layout), key=lambda r: r['n'])
                    choices = styles if scope != 'count' else [('count_input', 'Input queries', '#2878b5', 'o'), ('count_spatial', 'Spatial queries', '#299464', 's')]
                    for phase, label, color, marker in choices:
                        label += ' · lattice' if layout == 'lattice' else ' · shuffled'
                        def times(r, p=phase): return r['public'][p]['wall_ms'] if scope == 'public' else r['bare'][p]['graph']['device_interval_ms']
                        work = (lambda r: r['edges'] - r['n']) if scope == 'public' else (lambda r: r['candidate_visits_per_grid_pass'])
                        draw(axes[index, 0], rows, times, label=label, color=color, marker=marker, dashed=layout != 'lattice')
                        draw(axes[index, 1], rows, times, count=work, scale=1e9, label=label, color=color, marker=marker, dashed=layout != 'lattice')
                axes[index, 0].set_title(f'Requested neighbors {degree}: ' + {'public': 'complete search wall time', 'collect': 'complete collection graph interval', 'count': 'count graph interval'}[scope])
                axes[index, 0].legend(fontsize=6.8, ncol=2)
                axes[index, 1].set_title('Useful neighbor-pair output rate' if scope == 'public' else 'Candidate distance-test rate')
                axes[index, 1].set_ylabel('Directed non-self pairs/s (billions)' if scope == 'public' else 'Candidate tests/s (billions)')
            finishAxes(axes, report['records'])
            fig.suptitle(f'Existing grid permutation for query work · {dim}D · float32 · int32 IDs/offsets · RTX PRO 6000 Blackwell', fontsize=11)
            path = args.output_dir / f'query-order-{dim}d-{scope}-2026-10-15.png'
            save(fig, path, 'Median and shaded IQR; particle arrays, original CSR rows and J order are preserved; no additional sort or reindex pass.\n' +
                 ('Search includes count, prefix, allocation/padding and complete I/J collection; grid construction excluded.' if scope == 'public' else 'Preallocated graph batches; collection includes the I-fill launch for split paths. Count and collect each test candidates once.'))
            manifest['figures'].append(str(path))
        for row in report['records']:
            times = {k: v['median_wall_ms'] for k, v in row['public'].items()}
            manifest['cases'].append({k: row[k] for k in ('n', 'dim', 'degree', 'requested_neighbors', 'layout', 'edges', 'storage')} | dict(
                wall_ms=times, spatial_over_input_split_speedup=times['input_split'] / times['spatial_split'],
                combined_over_original_speedup=times['input_fused'] / times['spatial_split'],
                useful_nonself_pairs_per_second={k: v['useful_nonself_pairs_per_second'] for k,v in row['public'].items()}))
    manifest['output_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for name in manifest['figures'] for p in (Path(name), Path(name).with_suffix('.svg'))}
    manifest['complete'] = True; args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(manifest["figures"])} figures and {len(manifest["cases"])} cases')


if __name__ == '__main__': main()
