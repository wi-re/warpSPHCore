"""Regenerate measured tables and audit the recorded experiment."""
import hashlib
import json
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from development_paths import coreDevelopmentPath

ROOT=Path(__file__).resolve().parents[2]
HERE=coreDevelopmentPath('experiments/index_width')
r=json.loads((HERE/'results.json').read_text())
assert r['complete'] and len(r['records'])==10
assert sum(len(row['operators']) for row in r['records'])==120
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in r['source_sha256'].items())
assert all(o['exact_outputs'] and o['exact_production_reference'] for row in r['records'] for o in row['operators'])
assert all(all(row['checks'].values()) for row in r['records'])
for info in r['generated_sources'].values():
    assert hashlib.sha256(Path(info['generated_path']).read_bytes()).hexdigest()==info['generated_sha256']
header=Path(torch.__file__).resolve().parent/'include/ATen/native/TensorAdvancedIndexingUtils.h'
text=header.read_text()
assert 'indice = indice.to(at::kLong);' in text
large=[row for row in r['records'] if row['n']==4194304]
lines=['# Measured index-width results','',
       'RTX PRO 6000 Blackwell; Torch '+r['torch']+'; Warp '+r['warp']+'.',
       'Times below are warmed CUDA graph device medians, in ms. Seven samples, eight calls per batch.',
       'GB means decimal GB. Collection includes I and J writes and excludes other search stages.','',
       '## Collection and memory','',
       '| N | Mean entries | Input | I+J GB, int64 → int32 | Collect ms, int64 → int32 | Speedup |',
       '|---:|---:|---|---:|---:|---:|']
def ms(t):return t['graph']['median_device_interval_ms']
for row in large:
    d=row['collect'];a,b=ms(d['64']),ms(d['32'])
    lines.append(f'| {row["n"]:,} | {row["degree"]:.2f} | {row["layout"]} | {row["pair_storage_bytes"]["64"]/1e9:.3f} → {row["pair_storage_bytes"]["32"]/1e9:.3f} | {a:.3f} → {b:.3f} | {a/b:.3f}× |')
lines+=['','## SPH operators at 4,194,304 particles','',
        'Each lane count compares identical summation order between widths. Rates exclude self entries.',
        'A higher speedup means int32 is faster; throughput changes by the same ratio.','',
        '| Mean entries / input | Operation | Lanes | ms int64 | ms int32 | Speedup | Billion interactions/s int64 → int32 |',
        '|---|---|---:|---:|---:|---:|---:|']
speedups=[]
for row in large:
    for name in ('density','gradient','laplacian'):
        for lanes in (1,32):
            ops={o['bits']:o for o in row['operators'] if o['operation']==name and o['lanes']==lanes}
            a,b=(ops[bits]['timing']['median_graph_device_ms_per_call'] for bits in (64,32))
            rates=[ops[bits]['nonself_interactions_per_second']/1e9 for bits in (64,32)]
            speedups.append(a/b)
            lines.append(f'| {row["degree"]:.2f} / {row["layout"]} | {name} | {lanes} | {a:.3f} | {b:.3f} | {a/b:.3f}× | {rates[0]:.2f} → {rates[1]:.2f} |')
lines+=['','## Torch access and conversion','',
        'Scalar field indexed once by every J entry. These access timings include the output allocation in the captured action.',
        'Conversion device times use preallocated outputs; `.long()` wall time includes allocation and synchronization.','',
        '| Mean entries / input | Access | ms int64 | ms int32 | int64/int32 |',
        '|---|---|---:|---:|---:|']
for row in large:
    for name,d in row['torch_indexing'].items():
        a,b=ms(d['64']),ms(d['32'])
        lines.append(f'| {row["degree"]:.2f} / {row["layout"]} | {name} | {a:.3f} | {b:.3f} | {a/b:.3f}× |')
lines+=['','| Mean entries / input | J cast device ms | I+J cast device ms | Allocating J cast wall ms |',
        '|---|---:|---:|---:|']
for row in large:
    c=row['casts']
    lines.append(f'| {row["degree"]:.2f} / {row["layout"]} | {ms(c["j32_to64_preallocated"]):.3f} | {ms(c["ij32_to64_preallocated"]):.3f} | {c["j32_to64_allocating_wall"]["median_wall_ms"]:.3f} |')
lines+=['','## Compatibility probes','',
        '| Operation | CPU int32 | CUDA int32 |','|---|---|---|']
for name in r['torch_compatibility']['cpu']:
    flags=['accepted, exact' if r['torch_compatibility'][device][name]['int32'] else 'rejected' for device in ('cpu','cuda')]
    lines.append(f'| {name} | {flags[0]} | {flags[1]} |')
lines+=['','The installed Torch advanced-indexing header explicitly performs the int32→int64 cast.',
        f'Header: `{header}`; line {text[:text.index("indice = indice.to(at::kLong);")].count(chr(10))+1}.',
        f'SHA256: `{hashlib.sha256(header.read_bytes()).hexdigest()}`.','',
        'All recorded production/experiment/generated-source hashes, ten geometry checks and 120 exact operator checks passed audit.',
        'Timing samples are observations from one sweep, not statistical confidence intervals.',
        'See [scope, caveats, bounds and implementation sequence](README.md).','']
(HERE/'summary.md').write_text('\n'.join(lines))
(HERE/'audit.json').write_text(json.dumps(dict(complete=True,geometries=10,exact_operator_variants=120,
    production_and_experiment_hashes_match=True,generated_hashes_match=True,
    torch_advanced_indexing_header=dict(path=str(header),sha256=hashlib.sha256(header.read_bytes()).hexdigest()),
    largest_case_operator_speedup_range=[min(speedups),max(speedups)]),indent=2)+'\n')
print('Audit passed: 10 geometries, 120 exact operator variants; source hashes verified.')
