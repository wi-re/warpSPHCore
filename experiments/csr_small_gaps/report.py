#!/usr/bin/env python3
"""Summarize paired large-fixture results and the hardware-counter evidence."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
args=parser.parse_args()
p=args.archive
profiles=[]
for row in csv.DictReader((p/'profile.csv').open()):
    if not row.get('ID'):
        continue
    profiles.append(dict(path=row['Kernel Name'].split('/')[0],
        traversal_ms=float(row['gpu__time_duration.sum'])/1e6,
        dram_gb=float(row['dram__bytes.sum'])/1e9,l2_gb=float(row['lts__t_bytes.sum'])/1e9,
        registers=int(row['launch__registers_per_thread']),
        occupancy_pct=float(row['sm__warps_active.avg.pct_of_peak_sustained_active']),
        warp_instructions=float(row['smsp__inst_executed.sum']),
        long_scoreboard_pct=float(row['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']),
        short_scoreboard_pct=float(row['smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct'])))
(p/'profile-summary.json').write_text(json.dumps(profiles,indent=2)+'\n')
paths=['native32_public','delta_public','cell_public','band_public','nibble_public']
labels=['CSR32','Byte delta','Cell relative','Two-bit gaps','Nibble, 32 entries']
colors=['#222222','#2166ac','#67a9cf','#d95f02','#7570b3']
lines=['# GPU full-neighbor unity / small-gap codecs', '',
'Actual GPU encoders and fused interpolation consumers, isolated from production dispatch and AD.', '',
       f'Artifact archive: `{p}`. The throughput, storage and memory/throughput PNGs live there alongside raw JSONs.', '',
'## Scope and validation', '',
'* RTX PRO 6000 Blackwell Workstation, sm_120; PyTorch 2.13.0+cu130, CUDA 13 environment.',
'* Main sweep: 36 fresh-process fixtures, 1M/4M/16M particles, 1D/2D/3D, FP32/FP64, n_h=4/2.57.',
'* Schedules screened at 262,144 particles and frozen before held-out measurements.',
'* Nine randomized paired batches; CUDA events and wall times retained. Packing is included. Search, sorting, conversion, compression and correctness checks are excluded.',
'* Resident-buffer private controls and allocating wrappers reported separately. Native Torch wrappers are context for production call costs.',
'* Full correctness: 4,656 configurations, independent CPU readers, exact GPU index round trips, dense references, bit-exact matched raw controls, and a nondefault stream.',
'* Quick sanitizer matrix: 240 configurations; memcheck zero errors, racecheck zero errors/warnings.',
'* Main sweep plus four compact-layout fixtures and two rescreened bit-mask candidate fixtures: 42 held-out cases, all compute-process snapshots uncontended.',
'* All compared layouts coexist during measurement; peak Torch allocation is not the minimum memory needed by one layout.', '',
'## Actual large-case results', '',
'Below are allocating calls on 16M particles in 3D. Each table is one paired compact-layout run. GB are decimal, retained topology only, including counts/offsets/padding. The raw CSR graph has 4.305 billion edges at n_h=4 and 1.154 billion at n_h=2.57.', '']
fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
for ci,precision in enumerate(('float32','float64')):
    for ri,nh in enumerate((4,2.57)):
        case=json.loads((p/'compact'/f'{precision}-d3-h{nh:g}-n16000000.json').read_text())['cases'][0]
        records={r['name']:r for r in case['records']}
        storage={r['format']:sum(r['storage'].values()) for r in case['planning']}
        storage['native32']=case['csr32_bytes']
        native=records['native32_public']['event_ms']
        lines.extend([f'### {precision}, n_h={nh:g}, {case["mean_neighbors"]:.3f} neighbors per particle', '',
                      '| Layout | Time (ms) | Retained GB | Bytes/neighbor |',
                      '|---|---:|---:|---:|'])
        ax=axes[ri,ci]
        for name,label,color in zip(paths,labels,colors):
            size=storage[name.removesuffix('_public')]
            timing=records[name]['event_ms']
            lines.append(f'| {label} | {timing:.3f} | {size/1e9:.3f} | {size/case["edges"]:.3f} |')
            x,y=size/case['edges'],100*native/timing
            ax.scatter(x,y,s=70,color=color,label=label,marker='s' if name=='native32_public' else 'o')
        speedcase=json.loads((p/f'{precision}-d3-h{nh:g}-n16000000.json').read_text())['cases'][0]
        speedrecords={r['name']:r for r in speedcase['records']}
        speedplan=next(r for r in speedcase['planning'] if r['format']=='nibble')
        x=sum(speedplan['storage'].values())/speedcase['edges']
        y=100*speedrecords['native32_public']['event_ms']/speedrecords['nibble_public']['event_ms']
        ax.scatter(x,y,s=85,color=colors[-1],marker='x',label='Nibble, speed choice')
        ax.axhline(100,color='#888888',linestyle=':',linewidth=1)
        ax.set(title=f'{precision}, 3D, n_h={nh:g}',xlabel='Retained bytes per neighbor (smaller is better)',
               ylabel='Throughput relative to paired native CSR32 (%)',xlim=(0,4.8),ylim=(0,125))
        ax.grid(alpha=.2)
        lines.append('')
handles,legend=axes[0,0].get_legend_handles_labels()
fig.legend(handles,legend,loc='outside lower center',ncol=3)
fig.suptitle('16M particles: actual memory / throughput tradeoffs\nSpeed-choice nibble point uses its own paired CSR32 baseline')
fig.savefig(p/'memory-throughput.png',dpi=180);plt.close(fig)
lines.extend(['## Interpretation', '',
'The dense FP32 compact run reduces CSR32 storage by 86.9% with two-bit gaps and 89.9% with 32-entry nibble segments. Those calls cost 18.1% and 36.1% more time than paired native CSR32. Compression remains a capacity/throughput tradeoff here.', '',
'The faster nibble configuration selected 16-entry segments for dense FP32: 0.541 B/neighbor and 24.899 ms in the main run, compared with paired CSR32 at 18.595 ms. The fixed 32-entry compact configuration reaches 0.407 B/neighbor, close to the earlier 0.401 storage-only model. Dummy first controls and four-byte row alignment explain the small difference. Band reaches 0.529 versus the earlier 0.522 model.', '',
'For dense FP64, the speed-selected layouts are all approximately 253–256 ms versus CSR32 at 253.732 ms. The more compact 32-entry nibble variant costs more: 281.369 ms versus paired CSR32 at 252.673 ms. Precision alone therefore does not guarantee negligible decode overhead; scheduling and register lifetimes still matter.', '',
'At 16M in 1D with n_h=4, two-bit gaps take 0.923 ms versus paired native CSR32 at 1.257 ms, about 36% more throughput. In 2D, byte delta is the strongest of these compressed choices at that size. No single codec wins across dimensions and precision.', '',
'## Profile: 1M, 3D, n_h=4, FP32', '',
'These are traversal-only counters from three-pass hardware profiling. They exclude packing and wrapper costs and must not be substituted for ordinary whole-call measurements. Caches and clocks were uncontrolled consistently across paths; ordinary randomized event timings remain the performance comparison.', '',
'| Path | Traversal ms | DRAM GB | L2 GB | Registers | Occupancy % | Billion warp instructions |',
'|---|---:|---:|---:|---:|---:|---:|'])
for row in profiles:
    lines.append(f'| {row["path"]} | {row["traversal_ms"]:.3f} | {row["dram_gb"]:.3f} | {row["l2_gb"]:.3f} | {row["registers"]} | {row["occupancy_pct"]:.1f} | {row["warp_instructions"]/1e9:.3f} |')
lines.extend(['',
'Two-bit and nibble consumers cut DRAM traffic by about 83% relative to matched private CSR32, but execute 53% and 99% more warp instructions. Band uses 72 registers and 55% occupancy versus raw at 61 and 62%; nibble reaches similar occupancy to raw yet remains slower. Thus occupancy is not the sole problem. Prefix sums, payload addressing, byte/nibble unpacking and the row-carried cursor/ID dependencies replace memory work with integer and shuffle work. The lower long-scoreboard stall fraction is consistent with reduced DRAM waiting, not an overall speed improvement.', '',
'## Rejected decoder candidate', '',
'A second implementation reads two-bit control words and uses population counts to locate payloads instead of a subgroup length scan. It passed the same full correctness matrix. Rescreening selected a new schedule, but held-out FP32 dense results were worse: 1.459 ms at 1M and 24.203 ms at 16M, versus 1.411 and 22.424 ms for the simpler decoder. Native paired baselines stayed similar (about 1.15 and 18.6–18.9 ms). The candidate source and binary are archived; it is not an extra production or experiment-dispatch option.', '',
'## Costs and next boundary', '',
'Compression is currently built from an exact CSR graph and needs its own count/fill passes and a planning prefix sum. Those costs are in each JSON under planning.wall_ms. The row-per-thread encoder favors clear race-free packing; it has not been optimized. Neither direct compressed construction during radius search nor backward/adjoint consumers are implemented. CSR construction remains the peak-memory constraint until that boundary changes.', '',
'The next performance experiment would be modest restart segments with explicit absolute IDs/payload offsets, permitting independent segment decode and gathers. That spends some of the memory saving to break row-carried dependencies. The current byte layout already shows that more bytes can buy faster interpolation.', '',
'## Files', '',
'`summary.json` / `summary.csv`: 432 main-sweep path measurements. `compact/`: four separate fixed-32 nibble comparisons. `bitmask-*`: candidate screens and held-out comparisons. `profile-summary.json`: selected counters; the Nsight report/CSV retain raw results. `source/`: measured sources and rejected candidate. The audit and checksums identify all retained artifacts.', '',
'Literature: [Band et al., Compressed Neighbour Lists for SPH](https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf); [Thaler/Keller, 2026 preprint](https://arxiv.org/html/2602.19873v1). This experiment implements their codec ideas on exact particle lists; it does not reproduce the shared-cluster algorithm or claim their full-simulation performance.', ''])
(p/'report.md').write_text('\n'.join(lines))
print('Wrote report, profile summary and memory/throughput plot')
