"""Generate scaling graphs and summaries from the paired study artifacts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NAMES={0:'Density',1:'Interpolation',2:'Scalar gradient',3:'Vector Jacobian',
       4:'Covariance',5:'Normalized interpolation',6:'CRK moments'}
COLORS={'template32':'#1677b8','direct32':'#29863a','template64':'#9553aa',
        'native_torch':'#e28420','warp':'#777777','cell_shared':'#c94845'}
LABELS={'template32':'Template CSR32','direct32':'Direct loop CSR32','template64':'Template CSR64',
        'native_torch':'Existing native/Torch','warp':'Warp Morton CSR64','cell_shared':'Template shared cells'}


def elapsed(record):return record.get('event_ms',record['wall_ms'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path)
    args=parser.parse_args();archive=args.archive
    fixtures=[json.loads(path.read_text()) for path in sorted((archive/'study').glob('*.json'))]
    fixtures=[item for item in fixtures if item.get('complete')]
    records=[]
    for fixture in fixtures:
        for record in fixture['records']:
            records.append(dict(record,**{name:fixture[name] for name in ('n','dimension','precision','nh')}))
    summary=dict(fixtures=len(fixtures),records=len(records),
        qualification=sum(len(item['qualification']) for item in fixtures),
        unqualified=[dict(n=item['n'],dimension=item['dimension'],precision=item['precision'],nh=item['nh'],**comparison)
                     for item in fixtures for comparison in item['qualification'] if not comparison['qualified']],
        nonfinite=sum(not comparison['finite'] for item in fixtures for comparison in item['qualification']),
        other_gpu_activity=sum(bool(value['other_processes']) for item in fixtures for value in item['activity']))
    for device in ('cpu','cuda'):
        lookup={}
        for record in records:
            if record['phase']=='resident_forward' and record['device']==device:
                lookup[(record['precision'],record['dimension'],record['n'],record['nh'],record['work'],record['path'])]=record
        ratios=[];spreads=[]
        for key,record in lookup.items():
            if key[-1]!='template32':continue
            direct=lookup[(*key[:-1],'direct32')]
            ratios.append(elapsed(record)/elapsed(direct))
            values=[elapsed(value) for value in record['samples']]
            spreads.append((max(values)-min(values))/statistics.median(values))
        summary[device+'_template_direct_ratio']=dict(
            median=float(np.median(ratios)),p10=float(np.percentile(ratios,10)),p90=float(np.percentile(ratios,90)),
            median_sample_range_fraction=float(np.median(spreads)))
    for precision in ('float32','float64'):
        figure,axes=plt.subplots(2,3,figsize=(13,7.8))
        handles={}
        for row,device in enumerate(('cuda','cpu')):
            for column,work in enumerate((1,2,6)):
                axis=axes[row,column]
                for path,color in COLORS.items():
                    selected=sorted([record for record in records if record['precision']==precision and record['dimension']==3
                        and record['nh']==4 and record['phase']=='forward' and record['device']==device
                        and record['work']==work and record['path']==path],key=lambda record:record['n'])
                    if not selected:continue
                    line,=axis.plot([record['n'] for record in selected],
                        [record['n']/elapsed(record)/1000 for record in selected],
                        marker='o',color=color,label=LABELS[path],linestyle='--' if path=='direct32' else '-')
                    handles[path]=line
                axis.set_xscale('log',base=2);axis.set_ylim(bottom=0)
                axis.set_title(('GPU' if device=='cuda' else 'CPU, 24 Torch threads')+' — '+NAMES[work])
                axis.set_xlabel('Particles');axis.set_ylabel('Million particles / second')
                axis.grid(alpha=.2)
        figure.suptitle(f'Initial template interface · {precision.upper()} · 3D · n_h=4\nAllocating calls; packing included; search/sort/transpose excluded')
        figure.legend(list(handles.values()),[LABELS[path] for path in handles],loc='lower center',ncol=3)
        figure.subplots_adjust(bottom=.16,top=.86,hspace=.45,wspace=.3)
        figure.savefig(archive/f'forward-scaling-{precision}.png',dpi=170,bbox_inches='tight');plt.close(figure)

        figure,axes=plt.subplots(2,3,figsize=(13,7.8))
        for row,device in enumerate(('cuda','cpu')):
            for column,work in enumerate((1,2,5)):
                axis=axes[row,column]
                for requested,color in [('all','#1677b8'),('fields','#e28420')]:
                    selected=[record for record in records if record['precision']==precision and record['dimension']==3
                        and record['nh']==4 and record['phase']=='vjp' and record['device']==device
                        and record['work']==work and record['requested']==requested]
                    by_size=defaultdict(dict)
                    for record in selected:by_size[record['n']][record['path']]=record
                    sizes=sorted(by_size)
                    axis.plot(sizes,[elapsed(by_size[n]['atomic'])/elapsed(by_size[n]['owned']) for n in sizes],
                        marker='o',color=color,label='All used inputs' if requested=='all' else 'Field inputs only')
                axis.axhline(1,color='#555555',linestyle=':',label='Equal time')
                axis.set_xscale('log',base=2);axis.set_ylim(bottom=0)
                axis.set_title(('GPU' if device=='cuda' else 'CPU, 24 Torch threads')+' — '+NAMES[work])
                axis.set_xlabel('Particles');axis.set_ylabel('Atomic time / owned time');axis.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        figure.legend(handles,labels,loc='lower center',ncol=3)
        figure.suptitle(f'CSR/transpose VJP ownership · {precision.upper()} · 3D · n_h=4\nAbove 1: owned traversal faster; below 1: atomic traversal faster')
        figure.subplots_adjust(bottom=.13,top=.86,hspace=.45,wspace=.3)
        figure.savefig(archive/f'vjp-ownership-{precision}.png',dpi=170,bbox_inches='tight');plt.close(figure)
    summary['records_data']=records
    (archive/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='records_data'},indent=2))


if __name__=='__main__':main()
