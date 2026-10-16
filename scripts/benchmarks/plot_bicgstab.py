#!/usr/bin/env python3
"""Paired whole-call timing and interaction-rate plots for BiCGStab."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    datasets, inputs, outputs = [], {}, {}
    for precision in ('float32','float64'):
        path = root/f'docs/regression/bicgstab_{precision}_2026_10_15.json'
        data = json.loads(path.read_text())
        for source,expected in data['sourceSha256'].items():
            assert hashlib.sha256((root/source).read_bytes()).hexdigest()==expected,source
        datasets.append(data)
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for mode in ('fixed','verified'):
        fig,axes = plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
        for row,data in enumerate(datasets):
            for engine,color,style,label in (
                ('torch','#777777','-','Torch'),('native','#2166ac','-','Native'),
                ('native-cached','#b2182b','-','Native cached'),
                ('native-out','#2166ac','--','Native / out'),
                ('native-cached-out','#b2182b','--','Native cached / out')):
                records = sorted([r for r in data['records'] if r['mode']==mode and r['engine']==engine and r['meanDegree']==57],key=lambda r:r['n'])
                x = [r['n'] for r in records]
                axes[row,0].plot(x,[r['medianWallMs'] for r in records],style,marker='o',color=color,label=label)
                axes[row,1].plot(x,[r['work']['directed_interactions_per_second']/1e9 for r in records],style,marker='o',color=color,label=label)
            target = '8 scheduled steps' if mode=='fixed' else f"rtol={records[0]['rtol']:g}"
            axes[row,0].set_title(f"{data['dtype']}, {target}",loc='left')
            axes[row,0].set_ylabel('Whole call (ms)')
            axes[row,1].set_ylabel('Directed interactions / s (billions)')
            for axis in axes[row]:
                axis.set_xscale('log',base=2)
                axis.set_yscale('log')
                axis.set_xticks([1024,65536,1048576],['1,024','65,536','1,048,576'])
                axis.set_xlabel('Particles')
                axis.grid(True,which='both',alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"BiCGStab on nonsymmetric SPH, 57 neighbors: {mode} work\n"
                     'Warmed whole calls; work includes frozen callbacks and residual verification',fontsize=12)
        for suffix in ('.png','.svg'):
            path = root/f'docs/development/assets/bicgstab-{mode}-2026-10-15{suffix}'
            fig.savefig(path,dpi=170)
            outputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
    manifest = dict(inputs=inputs,outputs=outputs,plotSourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'docs/regression/bicgstab_figures_2026_10_15.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved fixed/verified paired timing and interaction-rate figures.')


if __name__=='__main__':
    main()
