#!/usr/bin/env python3
"""Paired timing/rate figures and CSR amortization estimates from neighbor-search runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from plot_work_scaling import compact_n

STYLES = {'full_rebuild': ('Full rebuild', '#2878b5', 'o'),
          'search_prebuilt_grid': ('Search existing grid', '#d65b24', 's'),
          'count': ('Count kernel', '#2878b5', 'o'), 'collect': ('Collect kernel', '#d65b24', 's'),
          'grid_build': ('Eager grid', '#666666', 'o'),
          'fixed_grid_build': ('Fixed-buffer grid', '#2878b5', 's'),
          'cached_grid_build': ('Cached grid', '#299464', '^')}


def draw(ax, rows, timings, *, count=None, scale=1., label, color, marker, dashed=False):
    x = [r['n'] for r in rows]
    values = [np.asarray(timings(r), dtype=float) for r in rows]
    assert all(v.size >= 2 and np.isfinite(v).all() and (v > 0).all() for v in values)
    if count is not None:
        values = [count(r) * 1000 / v / scale for r, v in zip(rows, values)]
    median = [np.median(v) for v in values]
    lower = [np.quantile(v, .25) for v in values]
    upper = [np.quantile(v, .75) for v in values]
    ax.plot(x, median, color=color, marker=marker, linestyle='--' if dashed else '-', label=label, linewidth=1.8, markersize=4)
    ax.fill_between(x, lower, upper, color=color, alpha=.08)


def finishAxes(axes, rows):
    ns = sorted({r['n'] for r in rows})
    for a in axes.ravel():
        a.set_xscale('log')
        a.set_xticks(ns)
        a.xaxis.set_major_formatter(FuncFormatter(compact_n))
        a.set_xlabel('Particles N')
        a.grid(True, alpha=.22)
        a.set_autoscaley_on(True)
        a.autoscale_view(scalex=False, scaley=True)
        if a in axes[:, 0]:
            a.set_yscale('log')
            a.set_ylabel('Time (ms)')
        else:
            a.set_ylim(bottom=0)
        low, high = a.get_ylim()
        assert all(low <= y <= high for line in a.lines for y in line.get_ydata())
        a.tick_params(labelsize=8)


def save(fig, path, footer):
    fig.text(.5, .015, footer, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .955), h_pad=2.2, w_pad=2.)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.inputs]
    if not all(r['complete'] for r in reports):
        parser.error('complete measurements required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    manifest = dict(input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
                    helper_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ('plot_neighbor_search.py', 'plot_work_scaling.py', 'bench_krylov.py')},
                    figures=[], cases=[], interpretation='Useful pair output rate, candidate distance-test rate and particles indexed/s are distinct metrics.')
    for report in reports:
        dim = report['configuration']['dim']
        hardware = report['gpu'].removeprefix('NVIDIA ').removesuffix(' Workstation Edition')
        precision = report['dtype'].removeprefix('torch.')
        for requested in report['configuration']['neighbors']:
            selected = [r for r in report['records'] if r['requested_neighbors'] == requested]
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            for layout in report['configuration']['layouts']:
                rows = sorted([r for r in selected if r['layout'] == layout], key=lambda r: r['n'])
                dashed = layout != 'lattice'
                suffix = ' · jittered/shuffled' if dashed else ' · lattice'
                for phase in ('full_rebuild', 'search_prebuilt_grid'):
                    label, color, marker = STYLES[phase]
                    timing = lambda r, p=phase: r['public'][p]['wall_ms']
                    draw(axes[0, 0], rows, timing, label=label+suffix, color=color, marker=marker, dashed=dashed)
                    draw(axes[0, 1], rows, timing, count=lambda r:r['physical_nonself_pairs'], scale=1e9,
                         label=label+suffix, color=color, marker=marker, dashed=dashed)
                for phase in ('count', 'collect'):
                    label, color, marker = STYLES[phase]
                    timing = lambda r,p=phase:r['bare'][p]['graph']['device_interval_ms']
                    draw(axes[1, 0], rows, timing, label=label+suffix, color=color, marker=marker, dashed=dashed)
                    draw(axes[1, 1], rows, timing, count=lambda r:r['candidate_visits_per_grid_pass'], scale=1e9,
                         label=label+suffix, color=color, marker=marker, dashed=dashed)
                for phase in ('grid_build', 'fixed_grid_build', 'cached_grid_build'):
                    label, color, marker = STYLES[phase]
                    timing = lambda r,p=phase:r['public'][p]['wall_ms']
                    draw(axes[2, 0], rows, timing, label=label+suffix, color=color, marker=marker, dashed=dashed)
                    draw(axes[2, 1], rows, timing, count=lambda r:r['n'], scale=1e6,
                         label=label+suffix, color=color, marker=marker, dashed=dashed)
            for a,title in zip(axes[:,0], ['Public neighbor search: synchronized wall time',
                                         'Bare count/collect: graph device interval', 'Grid construction: synchronized wall time']):
                a.set_title(title)
                a.legend(fontsize=7, ncol=2)
            for a,title,ylabel in zip(axes[:,1], ['Useful neighbor-pair output throughput', 'Candidate distance-test throughput', 'Grid construction throughput'],
                                     ['Directed non-self pairs/s (billions)', 'Candidate tests/s (billions; includes self)', 'Particles indexed/s (millions)']):
                a.set_title(title)
                a.set_ylabel(ylabel)
            finishAxes(axes, selected)
            fig.suptitle(f'{dim}D periodic neighbor search · requested degree {requested} · {precision} · {hardware}', fontsize=12)
            path = args.output_dir / f'neighbor-search-{dim}d-{requested}-2026-10-15.png'
            save(fig, path, 'Median and shaded IQR; solid = lattice, dashed = jittered/shuffled. Actual degrees vary and are recorded.\n'
                 'Public search counts and fills once each; graph rows isolate preallocated production kernels. Grid builds include input copies for fixed/cached paths.')
            manifest['figures'].append(str(path))
        for row in report['records']:
            entry = {k:row[k] for k in ('n','dim','layout','degree','requested_neighbors','edges','candidate_visits_per_grid_pass','candidate_to_accepted_ratio')}
            entry['wall_ms'] = {k:v['median_wall_ms'] for k,v in row['public'].items()}
            entry['bare_graph_ms'] = {k:v['graph']['median_device_interval_ms'] for k,v in row['bare'].items()}
            entry['amortization_estimates'] = []
            if 'laplacian' in row:
                for lanes in (1,32):
                    csr = next(x for x in row['laplacian'] if x['traversal']=='adjacency' and x['lanes']==lanes)
                    grid = next(x for x in row['laplacian'] if x['traversal']=='grid' and x['lanes']==lanes)
                    for scope,key in [('public_wall','median_eager_wall_ms'),('graph_device','median_graph_device_ms_per_call')]:
                        saving = grid['timing'][key] - csr['timing'][key]
                        conversion = row['public']['search_prebuilt_grid']['median_wall_ms']
                        entry['amortization_estimates'].append(dict(lanes=lanes, scope=scope,
                            grid_ms=grid['timing'][key], csr_ms=csr['timing'][key], conversion_wall_ms=conversion,
                            first_cheaper_pass_count=None if saving<=0 else math.floor(conversion/saving)+1,
                            model='same fixed geometry/field; grid build common to both; search wall plus K steady CSR calls vs K grid calls; excludes capture/trajectory costs'))
                for scope,key in [('public_wall','median_eager_wall_ms'),('graph_device','median_graph_device_ms_per_call')]:
                    csr=min((x for x in row['laplacian'] if x['traversal']=='adjacency'),key=lambda x:x['timing'][key])
                    grid=min((x for x in row['laplacian'] if x['traversal']=='grid'),key=lambda x:x['timing'][key])
                    saving=grid['timing'][key]-csr['timing'][key]
                    conversion=row['public']['search_prebuilt_grid']['median_wall_ms']
                    entry['amortization_estimates'].append(dict(lanes='best_measured_per_traversal',scope=scope,
                        csr_lanes=csr['lanes'],grid_lanes=grid['lanes'],grid_ms=grid['timing'][key],csr_ms=csr['timing'][key],
                        conversion_wall_ms=conversion,first_cheaper_pass_count=None if saving<=0 else math.floor(conversion/saving)+1,
                        model='minimum over measured lanes 1/32 only; common grid build cancels; indicative conversion-wall/steady-call estimate, not an end-to-end capture or trajectory timing'))
            manifest['cases'].append(entry)
        if dim==2 and all('laplacian' in r for r in report['records']):
            # Matched lattice operator panels; the raw record also includes disordered fields.
            degrees=report['configuration']['neighbors']
            fig, axes = plt.subplots(len(degrees),2,figsize=(12,4*len(degrees)),squeeze=False)
            for index, requested in enumerate(degrees):
                rows=sorted([r for r in report['records'] if r['layout']=='lattice' and r['requested_neighbors']==requested],key=lambda r:r['n'])
                for topology,lanes,color,marker in [('grid',1,'#666666','o'),('grid',32,'#a45ca6','v'),('adjacency',1,'#2878b5','s'),('adjacency',32,'#299464','^')]:
                    def timing(r,t=topology,l=lanes):
                        return next(o for o in r['laplacian'] if o['traversal']==t and o['lanes']==l)['timing']['graph_device_ms_per_call']
                    label=f'{topology}, {lanes} lanes'
                    draw(axes[index,0],rows,timing,label=label,color=color,marker=marker)
                    draw(axes[index,1],rows,timing,count=lambda r:r['physical_nonself_pairs'],scale=1e9,label=label,color=color,marker=marker)
                axes[index,0].set_title(f'Laplacian: requested degree {requested}, actual {rows[0]["degree"]:g}')
                axes[index,1].set_title('Interaction throughput')
                axes[index,1].set_ylabel('Directed non-self interactions/s (billions)')
                axes[index,0].legend(fontsize=8,ncol=2)
            finishAxes(axes,report['records'])
            fig.suptitle(f'Matched Laplacian traversal · periodic 2D lattice · {precision} · {hardware}',fontsize=12)
            path=args.output_dir/'neighbor-search-traversal-2026-10-15.png'
            save(fig,path,'Warmed batch-graph device intervals per call; median and shaded IQR. Exact same particles, support radii and fields.\nGrid construction and CSR conversion excluded here; both are measured separately. No production defaults changed.')
            manifest['figures'].append(str(path))
    manifest['output_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for name in manifest['figures'] for p in (Path(name),Path(name).with_suffix('.svg'))}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Saved {len(manifest["figures"])} paired figures and {len(manifest["cases"])} case summaries')


if __name__=='__main__':
    main()
