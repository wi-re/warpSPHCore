#!/usr/bin/env python3
"""Build before/after tables and scientific plots from retained measurements."""
import argparse
import csv
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
args=parser.parse_args()
archive=args.archive
records=[]
for precision in ['float32','float64']:
    before=json.loads((archive/f'paired-baseline-{precision}.json').read_text())
    after=json.loads((archive/f'paired-candidate-{precision}.json').read_text())
    assert before['complete'] and after['complete']
    assert before['library_sha256']==after['library_sha256']
    assert before['native_sha256']!=after['native_sha256']
    for a,b in zip(before['cases'],after['cases'],strict=True):
        for key in ['dimension','n_h','n','edges']:assert a[key]==b[key]
        ar={r['name']:r for r in a['records']}
        br={r['name']:r for r in b['records']}
        for bits in [32,64]:
            x,y=ar[f'native{bits}_public'],br[f'native{bits}_public']
            record={key:a[key] for key in ['dimension','n_h','n','edges']}
            record.update(precision=precision,index_bits=bits,before_ms=x['event_ms'],after_ms=y['event_ms'],
                          before_wall_ms=x['wall_ms'],after_wall_ms=y['wall_ms'],
                          speedup=x['event_ms']/y['event_ms'])
            amin,amax=min(s['event_ms'] for s in x['samples']),max(s['event_ms'] for s in x['samples'])
            bmin,bmax=min(s['event_ms'] for s in y['samples']),max(s['event_ms'] for s in y['samples'])
            record.update(speedup_low=amin/bmax,speedup_high=amax/bmin,
                          sample_ranges_overlap=not (amax<bmin or bmax<amin))
            records.append(record)
assert len(records)==48
profiles={}
for precision in ['float32','float64']:
    rows=list(csv.DictReader((archive/f'promoted-profile-{precision}.csv').open()))
    assert rows[0]['gpu__time_duration.sum']=='ns'
    data=[]
    for row in rows[1:]:
        def value(key):return float(row[key].replace(',',''))
        op=lambda name:value(f'smsp__sass_thread_inst_executed_op_{name}_pred_on.sum')
        data.append(dict(name=row['Kernel Name'],milliseconds=value('gpu__time_duration.sum')/1e6,
            fp32_flops=op('fadd')+op('fmul')+2*op('ffma'),
            fp64_flops=op('dadd')+op('dmul')+2*op('dfma'),
            dram_bytes=value('dram__bytes.sum'),l2_bytes=value('lts__t_bytes.sum'),
            registers=value('launch__registers_per_thread'),
            dram_pct=value('dram__throughput.avg.pct_of_peak_sustained_elapsed'),
            long_scoreboard_pct=value('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),
            occupancy_pct=value('sm__warps_active.avg.pct_of_peak_sustained_active')))
    profiles[precision]=data
summary=dict(records=records,profiles=profiles)
(archive/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (archive/'summary.csv').open('w') as output:
    writer=csv.DictWriter(output,fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
for row,precision in enumerate(['float32','float64']):
    for col,nh in enumerate([4.,2.57]):
        ax=axes[row,col]
        for offset,bits,color in [(-.18,64,'#2779b5'),(.18,32,'#d66b25')]:
            selected=[next(r for r in records if r['precision']==precision and r['n_h']==nh and r['n']==1000000 and r['dimension']==d and r['index_bits']==bits) for d in [1,2,3]]
            y=np.array([r['speedup'] for r in selected])
            low=np.array([r['speedup_low'] for r in selected])
            high=np.array([r['speedup_high'] for r in selected])
            bars=ax.bar(np.arange(3)+offset,y,width=.34,color=color,label=f'CSR{bits}')
            ax.errorbar(np.arange(3)+offset,y,yerr=[y-low,high-y],fmt='none',ecolor='#444444',capsize=3,lw=1)
            for bar,r in zip(bars,selected):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f"{r['speedup']:.2f}×",ha='center',fontsize=9)
        ax.axhline(1,color='#333333',linestyle='--',linewidth=1)
        ax.set_title(f'{precision.upper()}, n_h = {nh:g}')
        ax.set_xticks(range(3),['1D','2D','3D'])
        ax.set_ylim(0,2.05)
        ax.set_ylabel('Before time / after time (higher is faster)')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
axes[0,0].legend()
fig.suptitle('Native/Torch interpolation: 1M particles, fixed Morton order and neighbor lists\nComplete-call CUDA events; no search or index conversion in either timing')
fig.text(.5,.015,'Whiskers span ratios of observed min/max samples, not confidence intervals. Dashed line: equal performance.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.04,1,.92))
fig.savefig(archive/'native-speedup.png',dpi=170)
plt.close(fig)

# Separate graph replay from kernel scheduling: its main benefit is small calls.
graphs=json.loads((archive/'graphs-final.json').read_text())
assert graphs['complete']
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,n in zip(axes,[32768,1000000]):
    cases=[c for c in graphs['cases'] if c['n']==n]
    for offset,name,color in [(-.18,'ordinary','#2779b5'),(.18,'graph','#5b9b50')]:
        heights=[c['results'][name]['wall_ms']*1000 for c in cases]
        bars=ax.bar(np.arange(3)+offset,heights,width=.34,label=name,color=color)
        for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{bar.get_height():.1f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(range(3),['1D','2D','3D'])
    ax.set_title(f'N = {n:,}')
    ax.set_ylabel('Call time (µs; lower is faster)')
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend()
fig.suptitle('FP32 native interpolation with fixed topology: ordinary calls versus CUDA Graph replay')
fig.tight_layout(rect=(0,0,1,.92))
fig.savefig(archive/'graph-overhead.png',dpi=170)
plt.close(fig)
for r in records:
    if r['n']==1000000 and r['dimension']==3:print(r)
