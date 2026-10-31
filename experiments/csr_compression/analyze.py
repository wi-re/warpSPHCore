#!/usr/bin/env python3
"""Plot measured compression costs and throughput with linear throughput axes."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive', type=Path)
args = parser.parse_args()
all_rows = []
colors = dict(csr64='#777777',csr32='#222222',delta='#1675bd',cell='#df8420')
labels = dict(csr64='CSR64',csr32='CSR32',delta='Byte deltas',cell='Cell-relative bytes')
for precision in ('float32','float64'):
    report = json.loads((args.archive/f'final-{precision}.json').read_text())
    assert report['complete']
    assert report['uncontended_process_snapshots']
    cases = report['cases']
    for case in cases:
        records = {record['name']:record for record in case['records']}
        for record in case['records']:
            name = record['name']
            encoding = name.removesuffix('_public')
            storage = case.get(encoding+'_bytes')
            planning = next((p for p in case['planning'] if p['format']==encoding),None)
            if planning:
                storage = sum(planning['storage'].values())
            all_rows.append(dict(precision=precision,dimension=case['dimension'],nh=case['nh'],
                particles=case['particles'],neighbors=case['mean_neighbors'],path=name,
                event_ms=record['event_ms'],wall_ms=record['wall_ms'],
                gpairs_s=case['edges']/record['event_ms']/1e6,
                vs_native32=records['native32_public']['event_ms']/record['event_ms'],
                vs_native64=records['native64_public']['event_ms']/record['event_ms'],
                storage_bytes=storage,bytes_per_edge=storage/case['edges'] if storage else None,
                plan_ms=statistics.median(planning['wall_ms']) if planning else None,
                max_abs=record['max_abs']))
    for public in (False,True):
        fig,axes = plt.subplots(2,3,figsize=(14,7.2),sharex=True)
        for row,nh in enumerate((4.,2.57)):
            for column,dimension in enumerate((1,2,3)):
                ax=axes[row,column]
                selected = sorted((c for c in cases if c['nh']==nh and c['dimension']==dimension),key=lambda c:c['particles'])
                sizes = [c['particles'] for c in selected]
                for encoding in ('csr64','csr32','delta','cell'):
                    name = (('native64_public' if encoding=='csr64' else 'native32_public')
                            if public and encoding in ('csr64','csr32') else encoding+('_public' if public else ''))
                    timings = [next(r for r in c['records'] if r['name']==name) for c in selected]
                    rates = [c['edges']/r['event_ms']/1e6 for c,r in zip(selected,timings)]
                    low = [v-c['edges']/max(s['event_ms'] for s in r['samples'])/1e6 for v,c,r in zip(rates,selected,timings)]
                    high = [c['edges']/min(s['event_ms'] for s in r['samples'])/1e6-v for v,c,r in zip(rates,selected,timings)]
                    ax.errorbar(sizes,rates,yerr=[low,high],color=colors[encoding],label=labels[encoding],
                                marker='o',markersize=4,capsize=2,linewidth=1.6)
                ax.set_xscale('log',base=2)
                ax.set_xticks(sizes,[f'{n/1000:g}k' if n<1000000 else '1M' for n in sizes])
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.2)
                ax.set_title(f'{dimension}D, n_h={nh:g}')
                if not column: ax.set_ylabel('Billion directed pairs / s')
                if row: ax.set_xlabel('Particles')
        handles,legend_labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='lower center',ncol=4,frameon=False)
        context='allocating calls; native/Torch raw controls' if public else 'matched resident buffers and CUDA launch path'
        fig.suptitle(f'{precision}: {context}\nIncludes field packing; excludes search, sorting and compression. Whiskers: observed batch range.')
        fig.tight_layout(rect=(0,.05,1,.91))
        fig.savefig(args.archive/f'throughput-{precision}-{"public" if public else "matched"}.png',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,dimension in zip(axes,(1,2,3)):
        cases_at_size=[c for c in cases if c['dimension']==dimension and c['particles']==1000000]
        for i,nh in enumerate((4.,2.57)):
            case=next(c for c in cases_at_size if c['nh']==nh)
            sizes=[case['csr64_bytes'],case['csr32_bytes']]+[
                sum(next(p for p in case['planning'] if p['format']==encoding)['storage'].values())
                for encoding in ('delta','cell')]
            ax.bar(np.arange(4)+(i-.5)*.35,np.array(sizes)/1e6,width=.35,label=f'n_h={nh:g}')
        ax.set_xticks(np.arange(4),['CSR64','CSR32','Deltas','Cell bytes'],rotation=15)
        ax.set_title(f'{dimension}D, 1M particles')
        ax.set_ylabel('Retained adjacency MB (decimal)')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle(f'{precision}: actual retained tensors, including padding and row metadata\nDifferent screened chunk sizes may be selected in each case.')
    fig.tight_layout(rect=(0,0,1,.88))
    fig.savefig(args.archive/f'storage-{precision}.png',dpi=180)
    plt.close(fig)
with (args.archive/'summary.csv').open('w') as file:
    writer=csv.DictWriter(file,fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
(args.archive/'summary.json').write_text(json.dumps(all_rows,indent=2)+'\n')
print(f'Wrote {len(all_rows)} result rows and six figures')
