#!/usr/bin/env python3
"""Paired complete-collection timing/throughput plots for split I/J experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_neighbor_search import draw,finishAxes,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--tag',default='2026-10-15',help='output suffix; use a distinct tag for a new baseline')
    args=parser.parse_args()
    reports=[json.loads(path.read_text()) for path in args.inputs]
    if not all(p['complete'] for p in reports): parser.error('completed measurements required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    manifest=dict(input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
                  helper_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('plot_split_collect.py','plot_neighbor_search.py','plot_work_scaling.py')},
                  figures=[],cases=[],scope='Matched fused collection versus J search plus I fill; both arrays present. Production split32 where reported, benchmark-only split128.')
    for report in reports:
        dim=report['configuration']['dim']; degrees=report['configuration']['neighbors']
        gpu=report['gpu'].removeprefix('NVIDIA ').removesuffix(' Workstation Edition')
        for scope in ('public','bare'):
            fig,axes=plt.subplots(len(degrees),2,figsize=(12,4*len(degrees)),squeeze=False)
            for index,degree in enumerate(degrees):
                for layout in report['configuration']['layouts']:
                    rows=sorted([r for r in report['records'] if r['requested_neighbors']==degree and r['layout']==layout],key=lambda r:r['n'])
                    for phase,label,color,marker in [('fused','Fused I+J','#666666','o'),('split32','J + I (32 lanes)','#2878b5','s'),('split128','J + I (128 lanes)','#299464','^')]:
                        dashed=layout!='lattice'; label+=' · jittered/shuffled' if dashed else ' · lattice'
                        def timing(r,p=phase): return r['public'][p]['wall_ms'] if scope=='public' else r['bare'][p]['graph']['device_interval_ms']
                        draw(axes[index,0],rows,timing,label=label,color=color,marker=marker,dashed=dashed)
                        draw(axes[index,1],rows,timing,count=lambda r:r['edges']-r['n'],scale=1e9,label=label,color=color,marker=marker,dashed=dashed)
                axes[index,0].set_title(f'Requested neighbors {degree}: '+('search wall time' if scope=='public' else 'complete collection graph interval'))
                axes[index,0].legend(fontsize=7,ncol=2)
                axes[index,1].set_title('Useful pair output throughput')
                axes[index,1].set_ylabel('Directed non-self pairs/s (billions)')
            finishAxes(axes,report['records'])
            index_dtype=report.get('index_dtype','torch.int64').removeprefix('torch.')
            offset_dtype=report.get('offset_dtype','torch.int32').removeprefix('torch.')
            fig.suptitle(f'Separate CSR I generation · {dim}D · {report["dtype"].removeprefix("torch.")} · IDs {index_dtype}, offsets {offset_dtype} · {gpu}',fontsize=11)
            path=args.output_dir/f'split-collect-{dim}d-{scope}-{args.tag}.png'
            save(fig,path,'Median and shaded IQR; identical output format and particle order. J-only timings are excluded: every plotted path produces both I and J.\n'+
                 ('Public search includes count, prefix, allocation/padding and collection; grid construction excluded.' if scope=='public' else 'Preallocated production collect versus identical J-only traversal plus cooperative row fill; extra launch included.'))
            manifest['figures'].append(str(path))
        for row in report['records']:
            times={k:v['median_wall_ms'] for k,v in row['public'].items()}
            bare={k:v['graph']['median_device_interval_ms'] for k,v in row['bare'].items()}
            manifest['cases'].append({k:row[k] for k in ('n','dim','degree','requested_neighbors','layout','edges')}|
                dict(public_ms=times,bare_graph_ms=bare,public_speedup32=times['fused']/times['split32'],
                     public_speedup128=times['fused']/times['split128'],collect_graph_speedup32=bare['fused']/bare['split32'],
                     collect_graph_speedup128=bare['fused']/bare['split128'],
                     i_rows32_useful_write_GB_per_second=(4 if report.get('index_dtype')=='torch.int32' else 8)*row['edges']/bare['i_rows32']/1e6,
                     index_dtype=report.get('index_dtype','torch.int64'),offset_dtype=report.get('offset_dtype','torch.int32'),
                     storage=row.get('storage'),
                     public_useful_nonself_pairs_per_second={k:(row['edges']-row['n'])*1000/v for k,v in times.items()},
                     public_candidate_distance_tests_per_second={k:2*row['candidate_visits_per_grid_pass']*1000/v for k,v in times.items()},
                     collect_candidate_distance_tests_per_second={k:row['candidate_visits_per_grid_pass']*1000/bare[k] for k in ('fused','split32','split128')}))
    manifest['output_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for name in manifest['figures'] for p in (Path(name),Path(name).with_suffix('.svg'))}
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Saved {len(manifest["figures"])} figures and {len(manifest["cases"])} cases')


if __name__=='__main__': main()
