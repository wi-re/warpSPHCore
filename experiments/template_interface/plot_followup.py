"""Graph the matched compact-record/finalization follow-up (all y axes start at zero)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def milliseconds(record):
    return record.get('event_ms',record['wall_ms'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path);args=parser.parse_args()
    fixtures=[json.loads(path.read_text()) for path in (args.archive/'packing').glob('*.json')]
    assert len(fixtures)==6 and all(item['complete'] for item in fixtures)
    for precision in ('float32','float64'):
        figure,axes=plt.subplots(1,2,figsize=(12,4.2))
        for axis,device in zip(axes,('cuda','cpu')):
            paths=[('volume_and_field','Initial record','#777777'),
                   ('weighted_field','Compact record','#1677b8'),
                   ('weighted_direct','Compact direct loop','#29863a'),
                   ('native_torch','Existing native/Torch','#e28420')]
            selected=sorted([item for item in fixtures if item['precision']==precision],key=lambda item:item['dimension'])
            x=np.arange(3);width=.19
            for offset,(path,label,color) in enumerate(paths):
                records=[next(row for row in item['records'] if row['device']==device
                    and row.get('resident') is False) for item in selected]
                values=[item['n']/milliseconds(row['results'][path])/1000 for item,row in zip(selected,records)]
                axis.bar(x+(offset-1.5)*width,values,width,label=label,color=color)
            axis.set_xticks(x,['1D','2D','3D']);axis.set_ylim(bottom=0)
            axis.set_title('GPU' if device=='cuda' else 'CPU · 24 Torch threads')
            axis.set_ylabel('Million particles / second');axis.grid(axis='y',alpha=.2)
        figure.suptitle(f'Final interpolation loader · {precision.upper()} · 1,048,576 particles · n_h=4\nPaired allocating calls, packing included; sorting/search excluded')
        handles,labels=axes[0].get_legend_handles_labels()
        figure.legend(handles,labels,loc='lower center',ncol=4)
        figure.subplots_adjust(bottom=.2,top=.8,wspace=.3)
        figure.savefig(args.archive/f'compact-interpolation-{precision}.png',dpi=170,bbox_inches='tight')
        plt.close(figure)

    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,precision in zip(axes,('float32','float64')):
        fixture=next(item for item in fixtures if item['precision']==precision and item['dimension']==3)
        rows=[row for row in fixture['records'] if row.get('phase')=='finalization_vjp']
        x=np.arange(2);width=.34
        for at,method in enumerate(('owned','atomic')):
            speedups=[]
            for requested in ('all','fields'):
                row=next(row for row in rows if row['requested']==requested)['results']
                speedups.append(milliseconds(row['per_edge_'+method])/milliseconds(row['per_query_'+method]))
            axis.bar(x+(at-.5)*width,speedups,width,label=method.title())
        axis.axhline(1,linestyle=':',color='#555555');axis.set_ylim(bottom=0)
        axis.set_xticks(x,['All used inputs','Field only']);axis.set_ylabel('Old time / final time')
        axis.set_title(precision.upper());axis.grid(axis='y',alpha=.2)
    figure.suptitle('Normalized interpolation VJP · GPU · 3D · 1M · n_h=4\nFinalizer pullback once per query rather than once per edge')
    handles,labels=axes[0].get_legend_handles_labels();figure.legend(handles,labels,loc='lower center',ncol=2)
    figure.subplots_adjust(bottom=.19,top=.78,wspace=.3)
    figure.savefig(args.archive/'finalization-vjp.png',dpi=170,bbox_inches='tight');plt.close(figure)


if __name__=='__main__':main()
