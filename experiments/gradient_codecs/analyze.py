#!/usr/bin/env python3
"""Plot the heavier-operator comparison with timing and accuracy kept separate."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
args=parser.parse_args();p=args.archive
reports=[json.loads(path.read_text()) for path in sorted(p.glob('float*-d*-h*-n*.json'))]
reports=[r for r in reports if r['complete']]
rows=[]
for report in reports:
    size={(r['work'],r['corrected'],r['format']):sum(r['storage'].values()) for r in report['planning']}
    for record in report['records']:
        fmt=record['path']
        if fmt in ('warp_unsorted','warp_sorted','csr64'):memory=report['csr64_bytes']
        elif fmt in ('native_torch_unsorted','native_torch_sorted','csr32'):memory=report['csr32_bytes']
        else:memory=size[record['work'],record['corrected'],fmt]
        rows.append(dict(precision=report['precision'],dimension=report['dimension'],nh=report['nh'],
            particles=report['particles'],edges=report['edges'],neighbors=report['mean_neighbors'],
            **record,retained_bytes=memory,bytes_per_neighbor=memory/report['edges'],
            giga_interactions_per_second=report['edges']/record['event_ms']/1e6))
(p/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
if not rows:raise SystemExit('No completed fixtures')
with (p/'summary.csv').open('w') as stream:
    columns=[key for key in rows[0] if key not in ('samples','configuration')]
    writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
paths=['warp_unsorted','warp_sorted','native_torch_sorted','csr64','csr32','delta','band','nibble']
labels={'warp_unsorted':'Warp CSR64, original order','warp_sorted':'Warp CSR64, Morton order',
        'native_torch_sorted':'Production native Torch CSR32','csr64':'Native prototype CSR64','csr32':'Native prototype CSR32',
        'delta':'Native byte delta','band':'Native two-bit gaps','nibble':'Native unity/nibble'}
colors=dict(zip(paths,('#888888','#7570b3','#222222','#66a61e','#1b9e77','#2166ac','#d95f02','#e7298a')))
work_labels={'gradient_scalar':'Scalar gradient','gradient_vector':'Vector Jacobian',
             'divergence':'Vector divergence','covariance':'Covariance'}
for precision in sorted({r['precision'] for r in rows}):
    for dimension in sorted({r['dimension'] for r in rows}):
        for nh in (4.,2.57):
            selected=[r for r in rows if r['precision']==precision and r['dimension']==dimension and r['nh']==nh]
            if not selected:continue
            for group,works in [('gradients',('gradient_scalar','gradient_vector')),
                                ('other',('divergence','covariance'))]:
                fig,axes=plt.subplots(2,2,figsize=(13,9))
                fig.subplots_adjust(left=.075,right=.99,bottom=.19,top=.83,hspace=.38,wspace=.20)
                for ri,work in enumerate(works):
                    for ci,corrected in enumerate((False,True)):
                        ax=axes[ri,ci]
                        for path in paths:
                            values=sorted((r for r in selected if r['work']==work and r['corrected']==corrected and r['path']==path),key=lambda r:r['particles'])
                            if values:ax.plot([r['particles']/1e6 for r in values],[r['giga_interactions_per_second'] for r in values],
                                'o--' if path in ('warp_unsorted','native_torch_sorted') else 'o-',color=colors[path],label=labels[path])
                        ax.set(title=f'{work_labels[work]} — {"CRK" if corrected else "uncorrected"}',
                            xlabel='Particles (millions)',ylabel='Billion exact interactions / s',ylim=(0,None))
                        ax.set_xscale('log',base=2)
                        ticks=(1,4,16) if dimension==3 else (1,4)
                        ax.set_xticks(ticks,[str(x) for x in ticks]);ax.grid(alpha=.25)
                handles,legend=axes[0,0].get_legend_handles_labels()
                fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,fontsize=9)
                warning='; FP32 1D periodic geometry comparisons may be unqualified' if precision=='float32' and dimension==1 else ''
                batch_note='\n16M Warp uses four query batches (concatenation included)' if dimension==3 and nh==4 else ''
                fig.suptitle(f'{precision}, {dimension}D, n_h={nh:g}: allocating operator calls{warning}\nSearch, sorting, encoding and CRK-factor construction excluded{batch_note}',fontsize=12,y=.99)
                fig.savefig(p/f'throughput-{precision}-d{dimension}-h{nh:g}-{group}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # The largest 3D fixture makes comparisons among compute intensities easy to scan.
    largest=max(r['particles'] for r in rows if r['precision']==precision and r['dimension']==3)
    for nh in (4.,2.57):
        selected=[r for r in rows if r['precision']==precision and r['dimension']==3 and r['nh']==nh and r['particles']==largest]
        fig,ax=plt.subplots(figsize=(15,6),constrained_layout=True)
        works=list(work_labels)
        groups=[(work,corrected) for work in works for corrected in (False,True)]
        for pi,path in enumerate(paths):
            xs=[];ys=[]
            for gi,(work,corrected) in enumerate(groups):
                record=next((r for r in selected if r['path']==path and r['work']==work and r['corrected']==corrected),None)
                reference=next((r for r in selected if r['path']=='warp_sorted' and r['work']==work and r['corrected']==corrected),None)
                if record and reference:xs.append(gi+(pi-(len(paths)-1)/2)*.10);ys.append(reference['event_ms']/record['event_ms'])
            ax.bar(xs,ys,width=.095,color=colors[path],label=labels[path])
        ax.axhline(1,color='#555555',linestyle=':',linewidth=1)
        ax.set_xticks(range(8),[f'{work_labels[w]}\n{"CRK" if c else "uncorrected"}' for w,c in groups])
        ax.set(ylabel='Throughput relative to sorted Warp',ylim=(0,None),
            title=f'{precision}, 3D, n_h={nh:g}, {largest/1e6:g}M particles — allocating calls')
        ax.grid(axis='y',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3)
        fig.savefig(p/f'relative-{precision}-h{nh:g}.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Analyzed',len(reports),'fixtures and',len(rows),'operator/path comparisons')
