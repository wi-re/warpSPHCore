"""Summaries and zero-based throughput graphs for the public-API paired study."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


NAMES = {'density':'Density', 'interpolate':'Scalar interpolation',
    'interpolate_vector':'Vector interpolation', 'interpolate_matrix':'Matrix interpolation',
    'gradient':'Scalar gradient', 'gradient_vector':'Vector gradient', 'gradient_matrix':'Matrix gradient'}


def elapsed(record, device):
    return record['event_ms' if device=='cuda' else 'wall_ms']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.study.read_text())
    assert report['complete'] and all(case['complete'] for case in report['fixtures'])
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for case in report['fixtures']:
        for device in ('cuda','cpu'):
            lookup = {(row['operation'],row['path']):row for row in case['records'] if row['device']==device}
            for operation in NAMES:
                if (operation,'baseline') not in lookup:
                    continue
                baseline, candidate = (lookup[operation,path] for path in ('baseline','candidate'))
                spreads = []
                for row in (baseline,candidate):
                    samples = [elapsed(value,device) for value in row['samples']]
                    spreads.append((max(samples)-min(samples))/statistics.median(samples))
                pairs.append({**{key:case[key] for key in ('precision','dimension','n','nh','wide')},
                    'device':device,'operation':operation,'baseline_ms':elapsed(baseline,device),
                    'candidate_ms':elapsed(candidate,device),
                    'speedup':elapsed(baseline,device)/elapsed(candidate,device),
                    'sample_range_fraction':spreads})
    summary = dict(fixtures=len(report['fixtures']), comparisons=len(pairs),
        qualification=len([check for case in report['fixtures'] for check in case['qualification']]),
        largest_output_error=max(check['error'] for case in report['fixtures'] for check in case['qualification']),
        other_gpu_processes=sum(bool(check['other_processes']) for check in report['activity']))
    for device in ('cpu','cuda'):
        values = [pair for pair in pairs if pair['device']==device]
        summary[device] = dict(median_speedup=float(np.median([pair['speedup'] for pair in values])),
            median_sample_range=float(np.median([spread for pair in values for spread in pair['sample_range_fraction']])),
            large_regressions=[pair for pair in values if pair['n']>=262144 and pair['speedup']<.9])

    for precision in ('float32','float64'):
        for dimension in (1,2,3):
            figure, axes = plt.subplots(2,3,figsize=(13,7.5))
            handles = {}
            for row,device in enumerate(('cuda','cpu')):
                for column,operation in enumerate(('density','interpolate','gradient')):
                    axis = axes[row,column]
                    for nh,style in [(4.,'-'),(2.57,'--')]:
                        for path,color in [('baseline','#777777'),('candidate','#1677b8')]:
                            selected = sorted([pair for pair in pairs if pair['precision']==precision
                                and pair['dimension']==dimension and pair['device']==device
                                and pair['operation']==operation and pair['nh']==nh and not pair['wide']],key=lambda pair:pair['n'])
                            if not selected:
                                continue
                            label = ('Previous native' if path=='baseline' else 'Integrated native')+f' · n_h={nh:g}'
                            line, = axis.plot([pair['n'] for pair in selected],
                                [pair['n']/pair[path+'_ms']/1000 for pair in selected],
                                color=color,linestyle=style,marker='o',label=label)
                            handles[label] = line
                    axis.set_xscale('log',base=2)
                    axis.set_xticks([1024,32768,262144,1048576],['1K','32K','256K','1M'])
                    axis.set_ylim(bottom=0)
                    axis.set_title(('GPU' if device=='cuda' else 'CPU · 24 threads')+' — '+NAMES[operation])
                    axis.set_xlabel('Particles')
                    axis.set_ylabel('Million particles / second')
                    axis.grid(alpha=.2)
            figure.suptitle(f'Production native CSR32 · {precision.upper()} · {dimension}D\nPublic allocating calls; packing included; search/sort excluded')
            figure.legend(list(handles.values()),list(handles),loc='lower center',ncol=2)
            figure.subplots_adjust(bottom=.16,top=.86,hspace=.42,wspace=.3)
            figure.savefig(args.output/f'production-{precision}-d{dimension}.png',dpi=170,bbox_inches='tight')
            plt.close(figure)

    figure, axes = plt.subplots(2,2,figsize=(14,8))
    names = list(NAMES)
    short = ['Density','Interp S','Interp V','Interp M','Grad S','Grad V','Grad M']
    for row,device in enumerate(('cuda','cpu')):
        for column,precision in enumerate(('float32','float64')):
            axis = axes[row,column]
            selected = {pair['operation']:pair for pair in pairs if pair['precision']==precision
                and pair['dimension']==3 and pair['n']==1048576 and pair['nh']==4 and not pair['wide'] and pair['device']==device}
            x = np.arange(len(names))
            for offset,(path,color,label) in enumerate([('baseline','#777777','Previous native'),('candidate','#1677b8','Integrated native')]):
                axis.bar(x+(offset-.5)*.36,[selected[name][path+'_ms'] for name in names],.36,color=color,label=label)
            axis.set_xticks(x,short,rotation=20,ha='right')
            axis.set_ylim(bottom=0)
            axis.set_ylabel('Milliseconds per call')
            axis.set_title(('GPU' if device=='cuda' else 'CPU · 24 threads')+' — '+precision.upper())
            axis.grid(axis='y',alpha=.2)
    handles,labels = axes[0,0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='lower center',ncol=2)
    figure.suptitle('Production operators · 3D · 1,048,576 particles · n_h=4 · ≈269 neighbors\nS / V / M = scalar / vector / matrix input')
    figure.subplots_adjust(bottom=.13,top=.87,hspace=.44,wspace=.25)
    figure.savefig(args.output/'production-3d-1m.png',dpi=170,bbox_inches='tight')
    plt.close(figure)
    summary['pairs'] = pairs
    (args.output/'production-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='pairs'},indent=2))


if __name__=='__main__':
    main()
