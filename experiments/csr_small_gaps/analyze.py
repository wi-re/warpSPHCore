#!/usr/bin/env python3
"""Plot measured full-call throughput, retained storage, and planning costs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
args=parser.parse_args()
reports=[]
for path in sorted(args.archive.glob('float*-d*-h*-n*.json')):
    report=json.loads(path.read_text())
    if report['complete']:
        reports.append(report)
colors={'native64_public':'#777777','native32_public':'#222222','delta_public':'#2166ac',
        'cell_public':'#67a9cf','band_public':'#d95f02','nibble_public':'#7570b3'}
labels={'native64_public':'Native Torch CSR64','native32_public':'Native Torch CSR32',
        'delta_public':'Byte delta','cell_public':'Cell relative','band_public':'Two-bit gaps',
        'nibble_public':'Unity / nibble'}
rows=[]
for report in reports:
    for case in report['cases']:
        storage={p['format']:sum(p['storage'].values()) for p in case['planning']}
        storage.update(csr32=case['csr32_bytes'],csr64=case['csr64_bytes'])
        for record in case['records']:
            name=record['name']
            fmt=name.removesuffix('_public')
            if fmt.startswith('native'): fmt='csr'+fmt.removeprefix('native')
            rows.append(dict(precision=report['precision'],dimension=case['dimension'],nh=case['nh'],
                particles=case['particles'],edges=case['edges'],mean_neighbors=case['mean_neighbors'],path=name,
                event_ms=record['event_ms'],wall_ms=record['wall_ms'],
                giga_interactions_per_second=case['edges']/record['event_ms']/1e6,
                retained_bytes=storage[fmt],bytes_per_neighbor=storage[fmt]/case['edges'],
                config=record['configuration'],max_abs=record['max_abs']))
(args.archive/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
with (args.archive/'summary.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
for precision in sorted({r['precision'] for r in rows}):
    selected=[r for r in rows if r['precision']==precision]
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ri,nh in enumerate((4.,2.57)):
        for ci,d in enumerate((1,2,3)):
            ax=axes[ri,ci]
            for name,color in colors.items():
                values=sorted((r for r in selected if r['dimension']==d and r['nh']==nh and r['path']==name),key=lambda r:r['particles'])
                if values: ax.plot([r['particles']/1e6 for r in values],[r['giga_interactions_per_second'] for r in values],
                                   'o-',color=color,label=labels[name],linewidth=1.7)
            ax.set(title=f'{d}D, n_h={nh:g}',xlabel='Particles (millions)',ylabel='Billion exact interactions / s',ylim=(0,None))
            ax.set_xscale('log',base=2);ax.set_xticks((1,4,16),('1','4','16'));ax.grid(alpha=.25)
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=3)
    fig.suptitle(f'{precision}: allocating calls, including reference packing; search and encoding excluded')
    fig.savefig(args.archive/f'throughput-{precision}.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ri,nh in enumerate((4.,2.57)):
        for ci,d in enumerate((1,2,3)):
            ax=axes[ri,ci]
            for name,color in colors.items():
                values=sorted((r for r in selected if r['dimension']==d and r['nh']==nh and r['path']==name),key=lambda r:r['particles'])
                if values: ax.plot([r['particles']/1e6 for r in values],[r['bytes_per_neighbor'] for r in values],
                                   'o-',color=color,label=labels[name])
            ax.set(title=f'{d}D, n_h={nh:g}',xlabel='Particles (millions)',ylabel='Retained bytes / exact neighbor',ylim=(0,None))
            ax.set_xscale('log',base=2);ax.set_xticks((1,4,16),('1','4','16'));ax.grid(alpha=.25)
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=3)
    fig.suptitle(f'{precision}: actual allocations, including counts, offsets, controls, payload and padding')
    fig.savefig(args.archive/f'storage-{precision}.png',dpi=180);plt.close(fig)
print(f'Analyzed {len(reports)} fixtures / {len(rows)} measurements')
