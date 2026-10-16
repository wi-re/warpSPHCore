"""Timing and actual directed non-self interaction rates from the width sweep."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from development_paths import coreDevelopmentPath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

HERE=coreDevelopmentPath('experiments/index_width')
report=json.loads((HERE/'results.json').read_text())
assert report['complete']
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25})
folder=HERE/'figures';folder.mkdir(parents=True,exist_ok=True)
for name in ('density','gradient','laplacian'):
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharex=True)
    for col,requested in enumerate((55,220)):
        rows=sorted([r for r in report['records'] if r['requested_neighbors']==requested and r['layout']=='lattice'],key=lambda r:r['n'])
        for lanes in (1,32):
            for bits,color in ((64,'#d97706'),(32,'#1673ad')):
                ops=[next(o for o in r['operators'] if (o['operation'],o['bits'],o['lanes'])==(name,bits,lanes)) for r in rows]
                style='-' if lanes==1 else '--'
                label=f'int{bits}, {lanes} lane'+('s' if lanes>1 else '')
                axes[0,col].plot([r['n'] for r in rows],[o['timing']['median_graph_device_ms_per_call'] for o in ops],style,color=color,marker='o',label=label)
                axes[1,col].plot([r['n'] for r in rows],[o['nonself_interactions_per_second']/1e9 for o in ops],style,color=color,marker='o',label=label)
        axes[0,col].set_title(f'Mean {rows[-1]["degree"]:.0f} neighbor entries / particle')
        for ax in axes[:,col]:
            ax.set_xscale('log');ax.set_yscale('log');ax.xaxis.set_major_formatter(EngFormatter())
        axes[0,col].set_ylabel('Device time per operation (ms)')
        axes[1,col].set_ylabel('Billion non-self interactions / s')
        axes[1,col].set_xlabel('Particles')
    axes[0,0].legend(fontsize=9)
    fig.suptitle(f'{name.capitalize()}: int32 versus int64 CSR neighbor IDs',fontsize=15)
    fig.text(.5,.015,'2D periodic lattice, float32; same CSR-only physics and row order; CUDA graph batch median.\nRTX PRO 6000 Blackwell; self entries excluded from interaction rate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95))
    for extension in ('png','svg'):fig.savefig(folder/f'{name}.{extension}',dpi=170)
    plt.close(fig)
