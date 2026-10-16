#!/usr/bin/env python3
"""Reproducible timing, interaction-rate and storage figures for this experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from plot_neighbor_search import draw,finishAxes,save

COLORS={'csr':'#666666','neighbor-major':'#2878b5'}
LAYOUTS={'lattice':('o','-'),'jittered-shuffled':('s','--'),'support-tail':('^',':')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--build-paths',type=Path,help='final count-validated production-query build/pipeline measurements')
    args=parser.parse_args()
    reports=[json.loads(p.read_text()) for p in args.inputs]
    if not all(p['complete'] for p in reports):parser.error('completed sweeps required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    manifest=dict(inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),ROOT/'scripts/benchmarks/plot_neighbor_search.py',ROOT/'scripts/benchmarks/plot_work_scaling.py')},
                  figures=[],cases=[])
    for report in reports:
        dim=report['configuration']['dim'];degrees=report['configuration']['neighbors']
        hardware=report['gpu'].removeprefix('NVIDIA ').removesuffix(' Workstation Edition')
        fig,axes=plt.subplots(len(degrees),2,figsize=(12,4*len(degrees)),squeeze=False)
        for index,degree in enumerate(degrees):
            selected=[r for r in report['records'] if r['requested_neighbors']==degree]
            for layout in report['configuration']['layouts']:
                rows=sorted([r for r in selected if r['layout']==layout],key=lambda r:r['n'])
                if not rows:continue
                for storage in ('csr','neighbor-major'):
                    timing=lambda r,s=storage:r['search'][s]['wall_ms']
                    label=f'{storage} · {layout}'
                    marker,_=LAYOUTS[layout]
                    for ax,rate in zip(axes[index],(False,True)):
                        draw(ax,rows,timing,count=(lambda r:r['edges']-r['n']) if rate else None,scale=1e9,
                             label=label,color=COLORS[storage],marker=marker,dashed=layout!='lattice')
            axes[index,0].set_title(f'Requested degree {degree}: direct search wall time')
            axes[index,0].legend(fontsize=7,ncol=2)
            axes[index,1].set_title('Useful neighbor output throughput')
            axes[index,1].set_ylabel('Directed non-self pairs/s (billions)')
        finishAxes(axes,report['records'])
        fig.suptitle(f'Initial guarded prototype: {dim}D search · {hardware}',fontsize=12)
        path=args.output_dir/f'search-{dim}d.png'
        save(fig,path,'Diagnostic initial variant: per-edge overflow guard and row-major wrapped queries. Both layouts store int64 J only.\nUse the validated-build figures for final construction comparisons. Median/IQR; grid construction excluded.')
        manifest['figures'].append(str(path))
        # Storage curves include counts and metadata, but not shared particles/grid.
        fig,axes=plt.subplots(len(degrees),2,figsize=(12,3.7*len(degrees)),squeeze=False)
        for index,degree in enumerate(degrees):
            selected=[r for r in report['records'] if r['requested_neighbors']==degree]
            for layout,color in [('lattice','#666666'),('jittered-shuffled','#2878b5'),('support-tail','#d65b24')]:
                rows=sorted([r for r in selected if r['layout']==layout],key=lambda r:r['n'])
                if not rows:continue
                marker,_=LAYOUTS[layout]
                axes[index,1].plot([r['n'] for r in rows],[r['padding_ratio'] for r in rows],label=layout,color=color,marker=marker)
                for storage,style in [('csr','--'),('neighbor-major','-')]:
                    axes[index,0].plot([r['n'] for r in rows],[r['storage_bytes'][storage]/2**30 for r in rows],
                                       label=f'{storage} · {layout}',color=color,marker=marker,linestyle=style)
            axes[index,0].set_title(f'Requested degree {degree}: neighbor storage')
            axes[index,0].legend(fontsize=7,ncol=2)
            axes[index,1].set_title('Neighbor-major slots / live neighbor entries')
            axes[index,1].set_ylabel('N_max / N_mean')
        finishAxes(axes,report['records'])
        for a in axes[:,0]:a.set_yscale('linear');a.set_ylim(bottom=0);a.set_ylabel('Allocated neighbor data (GiB)')
        for a in axes[:,1]:a.set_ylim(bottom=0);a.legend(fontsize=8)
        fig.suptitle(f'{dim}D neighbor storage · int64 indices · no I array in either layout',fontsize=12)
        path=args.output_dir/f'storage-{dim}d.png'
        save(fig,path,'Neighbor-major allocation is exactly N × observed N_max; CSR-J is exactly E. Counts/metadata included, grid and particle fields excluded.\nSupport-tail: jittered/shuffled input with 1% of supports doubled; restricted to the documented stress subset.')
        manifest['figures'].append(str(path))
        for degree in degrees:
            selected=sorted([r for r in report['records'] if r['requested_neighbors']==degree and r['layout']=='lattice'],key=lambda r:r['n'])
            shown=[name for name in ('density','gradient','laplacian') if name in report['configuration']['operations']]
            if not shown:continue
            fig,axes=plt.subplots(len(shown),2,figsize=(12,3.6*len(shown)),squeeze=False)
            for index,name in enumerate(shown):
                for storage,lanes,color,marker in [('csr',1,'#666666','o'),('neighbor-major',1,'#2878b5','s'),('csr',32,'#299464','^'),('neighbor-major',32,'#a45ca6','v')]:
                    if lanes not in report['configuration']['lanes']:continue
                    def timing(r,s=storage,l=lanes,n=name):
                        return next(o for o in r['operators'] if o['operation']==n and o['storage']==s and o['lanes']==l)['timing']['graph_device_ms_per_call']
                    label=f'{storage}, {lanes} lane'+('s' if lanes!=1 else '')
                    draw(axes[index,0],selected,timing,label=label,color=color,marker=marker)
                    draw(axes[index,1],selected,timing,count=lambda r:r['edges']-r['n'],scale=1e9,label=label,color=color,marker=marker)
                axes[index,0].set_title(name.capitalize()+': graph device interval')
                axes[index,0].legend(fontsize=8,ncol=2)
                axes[index,1].set_title('Interaction throughput')
                axes[index,1].set_ylabel('Directed non-self interactions/s (billions)')
            finishAxes(axes,selected)
            fig.suptitle(f'{dim}D lattice operators · requested degree {degree} · {hardware}',fontsize=12)
            path=args.output_dir/f'operators-{dim}d-{degree}.png'
            save(fig,path,'Warmed graph intervals per matched primal backend action; median and shaded IQR. No build/transpose included.\nSame physics, fields, particle order and within-row summation order; only index addressing/range calculation changes.')
            manifest['figures'].append(str(path))
        for row in report['records']:
            summary={k:row[k] for k in ('n','dim','layout','degree','maximum','padding_ratio','storage_bytes','j_bytes','requested_neighbors')}
            summary['search_ms']={k:v['median_wall_ms'] for k,v in row['search'].items()}
            summary['search_speedup']=summary['search_ms']['csr']/summary['search_ms']['neighbor-major']
            summary['operators']=[]
            for name in report['configuration']['operations']:
                for lanes in report['configuration']['lanes']:
                    operators={o['storage']:o for o in row['operators'] if o['operation']==name and o['lanes']==lanes}
                    a,b=operators['csr'],operators['neighbor-major']
                    summary['operators'].append(dict(operation=name,lanes=lanes,
                        csr_graph_ms=a['timing']['median_graph_device_ms_per_call'],neighbor_major_graph_ms=b['timing']['median_graph_device_ms_per_call'],
                        csr_wall_ms=a['timing']['median_eager_wall_ms'],neighbor_major_wall_ms=b['timing']['median_eager_wall_ms'],
                        speedup=a['timing']['median_graph_device_ms_per_call']/b['timing']['median_graph_device_ms_per_call']))
            manifest['cases'].append(summary)
    if args.build_paths:
        builds=json.loads(args.build_paths.read_text())
        if not builds['complete'] or 'collector_policy' not in builds:parser.error('completed count-validated build paths required')
        manifest['inputs'][str(args.build_paths)]=hashlib.sha256(args.build_paths.read_bytes()).hexdigest()
        manifest['builds']=builds['records']
        primary={(r['dim'],r['n'],r['layout'],r['requested_neighbors']):r for report in reports for r in report['records']}
        styles=[('csr','CSR-J','#666666'),('direct','Direct neighbor-major','#2878b5'),
                ('rows','CSR + row conversion','#d65b24'),('tile','CSR + tiled conversion','#299464')]
        for dim in sorted({r['dim'] for r in builds['records']}):
            rows=[r for r in builds['records'] if r['dim']==dim]
            fig,axes=plt.subplots(2,2,figsize=(13,9))
            x=np.arange(len(rows));width=.19
            labels=[f'{r["degree"]:.1f} neighbors · {r["layout"]}\nN={r["n"]:,}' for r in rows]
            for i,(phase,title) in enumerate([('search','Complete construction'),('pipeline','Construction + one Laplacian')]):
                for j,rate in enumerate((False,True)):
                    ax=axes[i,j]
                    for z,(method,label,color) in enumerate(styles):
                        values=[]
                        for r in rows:
                            times=np.asarray(r[phase][method]['wall_ms'],dtype=float)
                            base=primary[(dim,r['n'],r['layout'],r['requested_neighbors'])]
                            values.append((base['edges']-base['n'])*1000/times/1e9 if rate else times)
                        med=np.array([np.median(v) for v in values]);lo=np.array([np.quantile(v,.25) for v in values]);hi=np.array([np.quantile(v,.75) for v in values])
                        ax.bar(x+(z-1.5)*width,med,width,label=label,color=color,alpha=.9,
                               yerr=np.stack((med-lo,hi-med)),capsize=2,error_kw={'linewidth':.7})
                    ax.set_xticks(x,labels,rotation=17,ha='right',fontsize=7)
                    ax.set_title(title+(': effective SPH throughput' if rate and phase=='pipeline' else ': useful pair throughput' if rate else ': wall time'))
                    ax.set_ylabel(('SPH interactions/s incl. build (billions)' if phase=='pipeline' else 'Neighbor output pairs/s (billions)') if rate else 'Time (ms)')
                    ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
                    if j==0:ax.legend(fontsize=7,ncol=2)
            fig.suptitle(f'{dim}D validated build paths · fixed particle order · median and IQR',fontsize=13)
            path=args.output_dir/f'validated-builds-{dim}d.png'
            save(fig,path,'Production query strides; capacity validated from count metadata before launch. Conversion paths include temporary CSR construction.\nPipeline uses the best previously measured 1/32-lane choice per format; rate credits Laplacian visits only, not search distance tests. Grid build excluded.')
            manifest['figures'].append(str(path))
    manifest['output_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for name in manifest['figures'] for p in (Path(name),Path(name).with_suffix('.svg'))}
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Saved {len(manifest["figures"])} figures and {len(manifest["cases"])} geometries')


if __name__=='__main__':main()
