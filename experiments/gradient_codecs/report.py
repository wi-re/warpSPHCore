#!/usr/bin/env python3
"""Write measured tables and qualifications for the heavier-operator study."""
import argparse
import json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive',type=Path)
args=parser.parse_args();p=args.archive
reports=[json.loads(path.read_text()) for path in sorted(p.glob('float*-d*-h*-n*.json'))]
reports=[r for r in reports if r['complete']]
rows=json.loads((p/'summary.json').read_text())
work_names={'gradient_scalar':'Scalar gradient','gradient_vector':'Vector Jacobian',
            'divergence':'Vector divergence','covariance':'Covariance'}
paths=['warp_unsorted','warp_sorted','native_torch_sorted','csr64','csr32','delta','band','nibble']
labels=['Warp original CSR64','Warp Morton CSR64','Production native CSR32','Prototype native CSR64',
        'Prototype native CSR32','Byte delta','Two-bit gaps','Unity/nibble']
lines=['# Heavier GPU operators: sorting, CRK and neighbor compression','',
'Primal scalar gradients, vector Jacobians, divergence and covariance, with and without CRK kernel-gradient correction. All timings below are allocating operator calls, including packing and wrapper dispatch. Search, sorting, conversion, encoding, CRK-factor construction and qualification are excluded.','',
f'Artifact archive: `{p}`. JSON/CSV files retain individual samples, accuracy checks, configurations and planning costs. Throughput PNGs use zero-based axes.','',
'## Scope and controls','',
f'* {len(reports)} completed held-out fixtures, {len(rows)} operator/path comparisons.',
'* 1M and 4M particles in 1D/2D/3D, plus 16M in 3D; FP32/FP64; n_h=4 and 2.57.',
'* RTX PRO 6000 Blackwell Workstation, sm_120; CUDA 13 native build, Warp 1.17.0, PyTorch 2.13.0+cu130.',
'* Wendland2, Gather, Difference fields and TrueAllToAll; random uniform periodic particle sets, uniform supports/mass/density, smooth periodic scalar/vector fields.',
'* Independent 262,144-particle screens freeze segment, adjacent-entry and batching choices before held-out sizes. Blocks contain 128 threads. Warp retains its existing 32-lane kernel.',
'* Warp uses CSR64. Both native CSR64 and CSR32 prototype controls are included so narrower IDs are not mistaken for a backend advantage.',
'* Production native/Torch rejects CRK corrections. Corrected native results are explicitly experimental primals; no Warp fallback is timed as native.',
'* Sorted-backend timings are randomized paired batches. Original/sorted phases run separately to avoid retaining two multi-billion-edge CSR64 graphs simultaneously.',
'* Particles, fields, volumes and all CRK coefficients follow the same permutation. The same correction coefficients are reused after sorting. Neighbor counts are checked exactly; operator permutation parity is sampled at up to 16,384 deterministic particles.','',
'## Correctness','',
'* 28,032 standalone configurations: independent dense FP64 oracles, nonsymmetric gradB, variable/zero supports and mixed periodicity; exact raw64/raw32/byte/two-bit/nibble results at matching schedules.',
'* A nondefault stream is included. Targeted sanitizer checks cover 160 configurations across all four operators, both correction modes and every format; memcheck zero errors, racecheck zero errors/warnings.',
'* An initial unfiltered sanitizer run was stopped during expensive instrumentation of thousands of unused specializations. It is not counted as a passed check. Targeted kernel filters and a representative schedule were then used.',
'* Query-batched Warp is bit-identical to full calls for factors and all eight operator cases in both precisions, checked separately before the oversized runs.',
'* Real apparent-volume CRK factors are constructed from Warp moments. Small fixtures verify equivalence of the dimensionless and physical coefficient solves where the physical limiter is inactive.','',
'## Actual CRK factors and scale','',
'The existing solver applies an absolute det(m2)<1e-14 cutoff. In a fixed unit box, physical m2 scales as h² and its determinant as h^(2D), so shrinking support can turn nearly every sufficiently large 3D fixture into an identity correction. This experiment normalizes moments and their derivatives by h, invokes the unchanged coefficient solver in batches, then transforms B, gradA and gradB back by 1/h, 1/h and 1/h². Support is held fixed during the coordinate derivative.','',
'The shared CRK setup runs on the original-order graph before permutation. Its recorded cost must not be interpreted as the cost of a sorted CRK setup or as a full-step benchmark.', '',
'This is an experimental preparation step, not a production solver change. It keeps the corrected workload physically nontrivial across scales. The measured CRK-factor setup omits the convenience wrapper’s unrelated diagnostic-density pass. Both setup wall cost and reproduction residuals are retained in each JSON.','',
'| Precision | D | n_h | N | Physical cutoff fraction | Actual identity fraction | CRK setup ms |',
'|---|---:|---:|---:|---:|---:|---:|']
# Put the principal, matched-width conclusions ahead of the long data tables.
findings=['## Findings','']
for precision in ('float32','float64'):
    selected=[r for r in rows if r['precision']==precision and r['dimension']==3 and r['nh']==4]
    if not selected:continue
    n=max(r['particles'] for r in selected)
    def values(work,corrected):
        return {r['path']:r for r in selected if r['particles']==n and r['work']==work and r['corrected']==corrected}
    scalar=values('gradient_scalar',False);jacobian=values('gradient_vector',True)
    original=scalar['warp_unsorted']['event_ms'];ordered=scalar['warp_sorted']['event_ms']
    warp=jacobian['warp_sorted']['event_ms'];raw64=jacobian['csr64']['event_ms'];raw32=jacobian['csr32']['event_ms']
    band=jacobian['band']['event_ms'];nibble=jacobian['nibble']['event_ms']
    findings.extend([f'**{precision}, {n/1e6:g}M, 3D, n_h=4:** scalar-gradient sorting changes Warp from {original:.2f} to {ordered:.2f} ms ({original/ordered:.2f}×). The CRK Jacobian takes {warp:.2f} ms in sorted Warp, {raw64:.2f} ms in the matching-CSR64 native prototype ({warp/raw64:.2f}× throughput), and {raw32:.2f} ms with native CSR32 ({warp/raw32:.2f}×). Thus narrower IDs account for only part of the native gain.', '',
        f'The same CRK Jacobian takes {band:.2f} ms with two-bit gaps ({100*(band/raw32-1):+.1f}% time versus native CSR32) and {nibble:.2f} ms with nibble lists ({100*(nibble/raw32-1):+.1f}%). Two-bit topology is {jacobian["band"]["retained_bytes"]/1e9:.3f} GB versus {jacobian["csr32"]["retained_bytes"]/1e9:.3f} GB for CSR32. Small timing differences should be treated as comparable performance, not reliable speedups.', ''])
findings.extend(['Field rank and CRK also change accumulator size, correction loads and selected schedules. Decode overhead therefore does not decline uniformly with more mathematical work. See the per-operation ratios below; some FP32 vector cases are nearly throughput-neutral while others retain a measurable penalty.', '',
'The experimental raw consumers also improve some uncorrected operations over existing production native/Torch. They pack reference volume/fields in both precisions, prepare query/kernel coefficients once and stage independent gathers. These results are not a production dispatch or AD change, and are specific to the stated Wendland2/Gather/Difference scope.', ''])
position=lines.index('## Scope and controls')
lines[position:position]=findings

for r in reports:
    c=r['crk_setup']
    lines.append(f'| {r["precision"]} | {r["dimension"]} | {r["nh"]:g} | {r["particles"]} | {c["physical_limiter_fraction"]:.6g} | {c["identity_fraction"]:.6g} | {c["wall_ms"]:.3f} |')
lines.extend(['','## Large 3D results','',
'At 16M particles, n_h=4 in 3D, Warp rejects the roughly 4.3-billion-entry array because each array dimension must fit signed int32. Those Warp results therefore use four query batches with shared full reference arrays; the timed operation includes concatenation. Planning/rebasing stays outside. Native uses one full graph call. Smaller graphs, including 16M at n_h=2.57, use ordinary single-call Warp. The original failure log and the adapter verification are retained.', '',
'Times are CUDA-event medians in milliseconds. A missing production-native corrected entry means unsupported, not zero time. Neighbor memory is retained topology only; CSR construction and simultaneous comparison layouts still determine peak device allocation.',''])
for precision in ('float32','float64'):
    for nh in (4.,2.57):
        selected=[r for r in rows if r['precision']==precision and r['dimension']==3 and r['nh']==nh]
        if not selected:continue
        n=max(r['particles'] for r in selected)
        lines.extend([f'### {precision}, n_h={nh:g}, {n/1e6:g}M particles','',
            '| Operation | CRK | '+' | '.join(labels)+' |',
            '|---|---|'+'---:|'*len(paths)])
        for work,label in work_names.items():
            for corrected in (False,True):
                values={r['path']:r for r in selected if r['particles']==n and r['work']==work and r['corrected']==corrected}
                lines.append('| '+label+' | '+('yes' if corrected else 'no')+' | '+
                    ' | '.join(f'{values[path]["event_ms"]:.3f}' if path in values else '—' for path in paths)+' |')
        lines.extend(['','Compression relative to the same experimental native CSR32 consumer:','',
            '| Operation | CRK | Byte speed ratio | Two-bit speed ratio | Nibble speed ratio | Byte GB | Two-bit GB | Nibble GB |',
            '|---|---|---:|---:|---:|---:|---:|---:|'])
        for work,label in work_names.items():
            for corrected in (False,True):
                values={r['path']:r for r in selected if r['particles']==n and r['work']==work and r['corrected']==corrected}
                base=values['csr32']['event_ms']
                lines.append('| '+label+' | '+('yes' if corrected else 'no')+' | '+
                    ' | '.join(f'{base/values[path]["event_ms"]:.3f}×' for path in ('delta','band','nibble'))+' | '+
                    ' | '.join(f'{values[path]["retained_bytes"]/1e9:.3f}' for path in ('delta','band','nibble'))+' |')
        lines.append('')
failures=[]
for r in reports:
    for q in r['qualification']:
        if not q['qualified']:failures.append(dict(precision=r['precision'],dimension=r['dimension'],nh=r['nh'],particles=r['particles'],**q))
(p/'unqualified-comparisons.json').write_text(json.dumps(failures,indent=2)+'\n')
lines.extend(['## Numerical limits','',
f'{len(failures)} recorded comparisons did not meet the standard cross-backend relative-L2 tolerance. See `unqualified-comparisons.json` for exact paths and errors. The benchmark permits only the inherited FP32 periodic 1D geometry discrepancy; unexpected disagreements in other cases stop the run.','',
'Warp’s periodic geometry performs (dx+L/2)-L/2 even for a very small in-range displacement. Native geometry preserves that displacement directly. At million-particle 1D scales, FP32 rounding can therefore be a material fraction of support. Timings for such configurations are retained but are not claimed to compare numerically equivalent outputs. Native raw/compressed equivalence is separately established by the independent dense suite.','',
'## Boundaries','',
'These are fixed-topology primals. There is no direct compressed radius-search builder, CRK backward/adjoint implementation, or production CRK dispatch added here. CRK-factor construction remains Warp/Torch preparation shared by every corrected consumer. Covariance’s Warp wrapper also produces a neighbor-count output even when the caller requests only the matrix; native primals produce the requested matrix directly.','',
'Source, binaries, checksums, GPU activity and the final commit are retained in the artifact audit. The implementation lives in `experiments/gradient_codecs/`; earlier codec/planning modules are reused rather than copied into production.',''])
lines.extend(['## Selected plots','',
    f'[FP32 3D gradients]({p}/throughput-float32-d3-h4-gradients.png) · [FP64 3D gradients]({p}/throughput-float64-d3-h4-gradients.png)', '',
    f'[FP32 divergence/covariance]({p}/throughput-float32-d3-h4-other.png) · [FP64 divergence/covariance]({p}/throughput-float64-d3-h4-other.png)', '',
    f'[FP32 relative throughput]({p}/relative-float32-h4.png) · [FP64 relative throughput]({p}/relative-float64-h4.png)', '',
    'The archive also contains corresponding n_h=2.57 and 1D/2D plots.', ''])
(p/'report.md').write_text('\n'.join(lines))
print('Wrote report for',len(reports),'fixtures;',len(failures),'explicitly unqualified comparisons')
