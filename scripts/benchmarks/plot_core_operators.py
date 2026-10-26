#!/usr/bin/env python3
"""One timing/throughput figure per operator from saved, qualified samples.

Accepts unified operator suite JSON plus legacy bench_sph_work and neighbor
lane JSON. Historical inputs are labeled with their filenames and never
relabeled as current measurements. No benchmark or device imports.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def normalize(document, label, timing='wall'):
    for row in document['records']:
        if row.get('status', 'qualified') != 'qualified':
            continue
        values = row.get('wall_ms') if timing == 'wall' else row.get('isolated', {}).get('graph_device_ms_per_call')
        if values is None:
            continue
        if len(values) < 2 or not all(math.isfinite(t) and t > 0 for t in values):
            raise ValueError('need at least two finite positive samples')
        median = statistics.median(values)
        if timing == 'wall' and not math.isclose(median, row['median_wall_ms'], rel_tol=1e-9):
            raise ValueError('median differs from raw samples')
        counts = row.get('counts', {})
        interactions = counts.get('evaluated_nonself_pairs', row.get('work', {}).get('evaluated_nonself_pairs'))
        matrices = counts.get('matrices')
        if matrices is None and row.get('work', {}).get('local_matrix_items', 0):
            # Legacy CRK has both a determinant and pinv; normalize one batch
            # of particle matrices, not twice N summed across stage categories.
            matrices = row['n']
        count = interactions if interactions is not None else matrices
        if count is None:
            raise ValueError('missing measured work count')
        if count < 0:
            raise ValueError('negative work count')
        metadata = {key: row.get(key, default) for key, default in (
            ('dim', 2), ('traversal', 'adjacency'), ('lanes', document.get('operator_neighbor_lanes', {}).get(row['operation'], document.get('configured_neighbor_lanes', 'default'))), ('form', 'plain'),
            ('kernel', 'Wendland2'), ('support', 'Gather'), ('gradient', 'Difference'), ('laplacian', 'Brookshaw'))}
        metadata['field_shape'] = row.get('field_shape', [metadata['dim']] if row['operation'] in ('divergence', 'curl') else [])
        metadata.update(degree=row.get('neighbors', row.get('requested_neighbors', document.get('requested_neighbors', '?'))),
                        dtype=document.get('dtype', 'unknown'), input=label, gpu=document.get('gpu', 'unknown'))
        yield dict(operation=row['operation'], n=row['n'], samples_ms=values, median_ms=median,
                   count=count, unit='interactions/s' if interactions is not None else 'matrices/s',
                   rate=count * 1000 / median, metadata=metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--timing', choices=['wall', 'graph'], default='wall')
    p.add_argument('--neighbors', nargs='+', type=int, help='filter requested neighbor targets for readable option comparisons')
    p.add_argument('--historical', action='store_true', help='explicitly label plots historical')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    groups = defaultdict(list)
    manifest = dict(timing=a.timing, historical=a.historical, inputs={}, figures=[])
    for path in a.inputs:
        doc = json.loads(path.read_text())
        if not doc.get('complete', False):
            raise ValueError(f'incomplete input: {path}')
        manifest['inputs'][str(path.resolve())] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                      source_sha256=doc.get('source_sha256'), gpu=doc.get('gpu'),
                                                      torch=doc.get('torch'), warp=doc.get('warp'))
        for row in normalize(doc, path.stem, a.timing):
            if a.neighbors and row['metadata']['degree'] not in a.neighbors:
                continue
            groups[(row['operation'], row['metadata']['dim'], row['metadata']['dtype'], row['unit'])].append(row)
    if not groups:
        raise ValueError('no qualified timing samples to plot')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for (op, dim, dtype, unit), rows in sorted(groups.items()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        series = defaultdict(list)
        for row in rows:
            key = json.dumps(row['metadata'], sort_keys=True)
            series[key].append(row)
        for key, points in series.items():
            points.sort(key=lambda x: x['n'])
            if len({point['n'] for point in points}) != len(points):
                raise ValueError('duplicate N in one series; split inputs/configuration explicitly')
            meta = json.loads(key)
            label = f"{meta['traversal']} L{meta['lanes']} k≈{meta['degree']} {meta['form']}"
            if len({tuple(v['metadata']['field_shape']) for v in rows}) > 1:
                label += ' field=' + str(meta['field_shape'])
            for option in ('kernel', 'support', 'gradient', 'laplacian'):
                if len({v['metadata'][option] for v in rows}) > 1:
                    label += ' ' + str(meta[option])
            x = [v['n'] for v in points]
            times = [v['median_ms'] for v in points]
            rates = [v['rate'] for v in points]
            low, high = np.array([np.quantile(v['samples_ms'], [.25, .75]) for v in points]).T
            line, = axes[0].plot(x, times, marker='o', label=label)
            axes[0].fill_between(x, low, high, color=line.get_color(), alpha=.12)
            axes[1].plot(x, rates, marker='o', color=line.get_color())
            counts = np.array([v['count'] for v in points]) * 1000
            axes[1].fill_between(x, counts / high, counts / low, color=line.get_color(), alpha=.12)
        for ax in axes:
            ax.set_xscale('log')
            ax.grid(alpha=.2)
            ax.set_xlabel('Particles N')
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Total public-call time (ms)' if a.timing == 'wall' else 'Graph device interval per call (ms)')
        axes[1].set_ylabel(unit)
        axes[1].set_ylim(bottom=0)
        fig.suptitle(f"{'Historical · ' if a.historical else ''}{op} · {dim}D · {dtype}")
        axes[0].legend(fontsize=max(5, 9 - len(series) // 8), loc='best')
        fig.text(.5, .035, 'Median and IQR; matched work/time scope. Local stages use matrices/s. ' + str(rows[0]['metadata']['gpu']), ha='center', fontsize=8)
        fig.text(.5, .012, 'Inputs: ' + ', '.join(p.name for p in a.inputs), ha='center', fontsize=7)
        fig.tight_layout(rect=(0, .05, 1, .94))
        stem = a.output_dir / f'{op}-{dim}d-{dtype.removeprefix("torch.")}-{a.timing}'
        for suffix in ('.png', '.svg'):
            fig.savefig(stem.with_suffix(suffix), dpi=160)
        plt.close(fig)
        manifest['figures'].append(dict(operation=op, png=str(stem.with_suffix('.png').resolve()),
                                       svg=str(stem.with_suffix('.svg').resolve()), points=rows))
    (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(manifest["figures"])} figures -> {a.output_dir}')


if __name__ == '__main__':
    main()
