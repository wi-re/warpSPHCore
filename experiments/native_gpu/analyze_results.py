#!/usr/bin/env python3
"""Analyze complete paired public sweeps; never imports Torch or runs workloads."""
import argparse,csv,json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
args=parser.parse_args()
operations=['search','density','interpolate','gradient','divergence','curl','laplacian','covariance']
runs={}
records={}
for stage in ['checkpoint','packed']:
    for precision in ['float32','float64']:
        path=args.directory/f'paired-{stage}-gpu-{precision}.json'
        run=json.loads(path.read_text())
        if not run['complete']:parser.error(f'Incomplete run: {path}')
        runs[stage,precision]=run
        for record in run['records']:
            key=(precision,record['dimension'],record['n'],record['operation'],record['backend'])
            if (stage,key) in records:parser.error(f'Duplicate case: {key}')
            records[stage,key]=record
rows=[]
for (stage,key),after in records.items():
    if stage!='packed' or key[-1]!='native_torch':continue
    before=records['checkpoint',key]
    warp=records['packed',key[:-1]+('warp',)]
    if before['edges']!=after['edges']:parser.error(f'Changed edge count: {key}')
    rows.append(dict(precision=key[0],dimension=key[1],n=key[2],operation=key[3],
        baseline_ms=before['median_ms'],packed_ms=after['median_ms'],warp_ms=warp['median_ms'],
        speedup_vs_baseline=before['median_ms']/after['median_ms'],
        speedup_vs_warp=warp['median_ms']/after['median_ms'],
        packed_min_ms=after['min_ms'],packed_max_ms=after['max_ms'],
        edges=after['edges'],correctness=after['correctness']))
with (args.directory/'paired-summary.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,FuncFormatter,NullFormatter
import numpy as np
fig,axes=plt.subplots(1,2,figsize=(13,5.7),constrained_layout=True)
series=[('checkpoint','native_torch','Native checkpoint','#9aabb9',-.23),
        ('packed','native_torch','Native packed / reordered','#e07820',0),
        ('packed','warp','Warp reference','#3977a8',.23)]
for axis,precision in zip(axes,['float32','float64']):
    for stage,backend,label,color,offset in series:
        data=[records[stage,(precision,2,1000000,op,backend)] for op in operations]
        ms=[r['median_ms'] for r in data]
        axis.barh(np.arange(8)+offset,ms,height=.21,color=color,label=label,
                  xerr=[[r['median_ms']-r['min_ms'] for r in data],
                        [r['max_ms']-r['median_ms'] for r in data]],
                  error_kw=dict(elinewidth=.7,capsize=1.5))
    ratios={r['operation']:r['speedup_vs_baseline'] for r in rows
            if r['precision']==precision and r['dimension']==2 and r['n']==1000000}
    axis.set_yticks(np.arange(8),[f'{op.capitalize()} ({ratios[op]:.2f}×)' for op in operations])
    axis.invert_yaxis();axis.set_xscale('log')
    axis.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel('Completed public-call wall time (ms, log scale)')
    axis.set_title('FP32' if precision=='float32' else 'FP64')
    axis.grid(axis='x',alpha=.2);axis.set_axisbelow(True)
fig.suptitle('1,000,000 particles · 2D · ≈33 neighbors per particle\n'
             'Parentheses: speedup over 0560de5; whiskers: sample min–max',fontsize=14)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.savefig(args.directory/'paired-million.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for row,op in enumerate(['search','interpolate']):
    for col,dimension in enumerate([2,3]):
        axis=axes[row,col]
        for stage,backend,label,color,_ in series:
            data=sorted([r for (s,k),r in records.items() if s==stage and k[0]=='float32'
                         and k[1]==dimension and k[3]==op and k[4]==backend],key=lambda r:r['n'])
            ns=[r['n'] for r in data]
            axis.plot(ns,[r['median_ms'] for r in data],marker='o',color=color,label=label)
            axis.fill_between(ns,[r['min_ms'] for r in data],[r['max_ms'] for r in data],color=color,alpha=.12)
        axis.set(xscale='log',yscale='log',xlabel='Particles',ylabel='Completed wall time (ms)',
                 title=f'{op.capitalize()} · {dimension}D · FP32')
        axis.grid(alpha=.2)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.savefig(args.directory/'fp32-scaling.png',dpi=160);plt.close(fig)
lines=['# Paired native GPU exploration','',
       '![Million-particle comparison](paired-million.png)','',
       '![FP32 scaling](fp32-scaling.png)','',
       'Medians and sample min/max are completed public-call times, including temporary allocation and preparation.',
       'No verification runs in timed execution. Search uses fresh index/count/scan/collect/sort.',
       'All matched native cases, including FP64 and small-N cases, are in paired-summary.csv.','',
       '## Activity and provenance','']
for (stage,precision),run in runs.items():
    pid=str(run.get('pid'))
    other=sum(any(line.split(',')[0].strip()!=pid for line in snapshot['processes'].get('stdout','').splitlines()[1:])
              for snapshot in run['activity'])
    lines.append(f'- {stage} {precision}: {len(run["records"])} records; {other}/{len(run["activity"])} snapshots show another compute process.')
lines+=['','Activity snapshots are samples, not a guarantee of exclusive GPU ownership. Desktop and clock noise remain.',
        'Artifact hashes/source revisions and every raw sample are retained in each paired JSON.','']
(args.directory/'paired-graphs.md').write_text('\n'.join(lines))
print(args.directory/'paired-graphs.md')
