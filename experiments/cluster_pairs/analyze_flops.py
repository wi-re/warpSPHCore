#!/usr/bin/env python3
"""Join deterministic instruction counters with separate ordinary event timings."""
import argparse
import csv
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive', type=Path)
args = parser.parse_args()
archive = args.archive
backends = ['native_csr', 'warp_csr', 'cell_shared', 'cluster_raw', 'cluster_compressed']
labels = ['Native CSR', 'Warp CSR', 'Cell shared', 'Cluster raw', 'Cluster compressed']
records, kernels = [], []

for precision in ['float32', 'float64']:
    timing = json.loads((archive/f'timing-{precision}.json').read_text())
    profile = json.loads((archive/f'profile-{precision}.json').read_text())
    assert timing['complete'] and profile['complete']
    assert timing['artifacts'] == profile['artifacts']
    rows = list(csv.DictReader((archive/f'counters-{precision}.csv').open()))
    units, rows = rows[0], rows[1:]
    assert units['gpu__time_duration.sum'] == 'ns'
    assert units['dram__bytes.sum'] == units['lts__t_bytes.sum'] == 'byte'
    groups = {}
    for row in rows:
        label, kernel_name = row['Kernel Name'].split('/', 1)
        match = re.fullmatch(r'(float32|float64)_d([123])_h([\d.]+)_(.+)', label)
        assert match and match[1] == precision
        def value(name):
            return float(row[name].replace(',', ''))
        operations = {op: value(f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum')
                      for op in ['fadd', 'fmul', 'ffma', 'dadd', 'dmul', 'dfma']}
        kernel = dict(precision=precision, dimension=int(match[2]), n_h=float(match[3]),
            backend=match[4], name=kernel_name, instructions=operations,
            fp32_flops=operations['fadd']+operations['fmul']+2*operations['ffma'],
            fp64_flops=operations['dadd']+operations['dmul']+2*operations['dfma'],
            profile_ms=value('gpu__time_duration.sum')/1e6,
            dram_bytes=value('dram__bytes.sum'), l2_bytes=value('lts__t_bytes.sum'),
            integer_instructions=value('smsp__sass_thread_inst_executed_op_integer_pred_on.sum'),
            sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'))
        kernels.append(kernel)
        groups.setdefault((kernel['dimension'], kernel['n_h'], kernel['backend']), []).append(kernel)
    for case, profiled_case in zip(timing['cases'], profile['cases'], strict=True):
        assert case['dimension'] == profiled_case['dimension']
        assert case['n_h'] == profiled_case['n_h'] and case['edges'] == profiled_case['edges']
        for backend in backends:
            selected = groups.pop((case['dimension'], case['n_h'], backend))
            record = {key: case[key] for key in ['precision', 'dimension', 'n_h', 'n', 'neighbors', 'edges']}
            record.update(backend=backend, qualified=case['parity'][backend], kernels=len(selected),
                          event_ms=case['timings'][backend]['event_ms'],
                          event_min_ms=min(case['timings'][backend]['samples_ms']),
                          event_max_ms=max(case['timings'][backend]['samples_ms']))
            for key in ['fp32_flops', 'fp64_flops', 'profile_ms', 'dram_bytes', 'l2_bytes', 'integer_instructions']:
                record[key] = sum(k[key] for k in selected)
            record['weighted_sm_clock_hz'] = sum(k['sm_clock_hz']*k['profile_ms'] for k in selected)/record['profile_ms']
            for kind in ['fp32', 'fp64']:
                record[f'{kind}_tflops_event'] = record[f'{kind}_flops']/record['event_ms']/1e9
                record[f'{kind}_tflops_profile'] = record[f'{kind}_flops']/record['profile_ms']/1e9
            kind = 'fp32' if precision == 'float32' else 'fp64'
            record['tflops_event'] = record[f'{kind}_tflops_event']
            record['tflops_profile'] = record[f'{kind}_tflops_profile']
            record['nominal_peak_tflops'] = 125 if precision == 'float32' else 125/64
            record['nominal_peak_pct'] = record['tflops_event']/record['nominal_peak_tflops']*100
            # 188 enabled SMs, 128 FP32 or 2 FP64 FMA lanes per SM.
            lanes = 128 if precision == 'float32' else 2
            record['profile_clock_peak_tflops'] = 188*lanes*2*record['weighted_sm_clock_hz']/1e12
            record['profile_clock_peak_pct'] = record['tflops_profile']/record['profile_clock_peak_tflops']*100
            record['flops_per_true_pair'] = record[f'{kind}_flops']/record['edges']
            records.append(record)
    assert not groups
assert len(records) == 60
(archive/'summary.json').write_text(json.dumps(dict(records=records, kernels=kernels), indent=2)+'\n')
with (archive/'summary.csv').open('w') as output:
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
for row, nh in enumerate([4., 2.57]):
    for col, dimension in enumerate([1, 2, 3]):
        ax = axes[row, col]
        for offset, precision, color in [(-.19, 'float32', '#2878b5'), (.19, 'float64', '#d35e22')]:
            selected = [next(r for r in records if r['precision']==precision and r['dimension']==dimension
                             and r['n_h']==nh and r['backend']==backend) for backend in backends]
            heights = [r['nominal_peak_pct'] if r['qualified'] else 0 for r in selected]
            bars = ax.bar(np.arange(5)+offset, heights, width=.36, color=color, label=precision.upper())
            for bar, result in zip(bars, selected):
                text = f"{result['nominal_peak_pct']:.1f}%" if result['qualified'] else 'N/Q'
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, text, ha='center', va='bottom', fontsize=8, rotation=90)
        ax.axhline(100, color='#555555', linestyle='--', linewidth=1)
        ax.set_title(f'{dimension}D, n_h = {nh:g}')
        ax.set_xticks(range(5), labels, rotation=25, ha='right')
        ax.set_ylim(0, 108)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel('Executed FLOP/s / nominal scalar peak (%)')
axes[0, 0].legend(loc='upper left')
fig.suptitle('Interpolation: achieved scalar arithmetic throughput (higher is more arithmetic per second)\n'
             '1M particles; full-call CUDA event time; FMA = 2 FLOPs; nominal peaks: FP32 125, FP64 1.953 TFLOP/s', fontsize=13)
fig.text(.5, .012, 'More arithmetic throughput does not imply faster interpolation. N/Q: known FP32 1D Warp periodic-rounding mismatch.', ha='center')
fig.tight_layout(rect=(0, .04, 1, .92))
fig.savefig(archive/'achieved-fraction-of-peak.png', dpi=170)
plt.close(fig)

# Show how much floating-point work the different algorithms actually execute.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, precision in zip(axes, ['float32', 'float64']):
    selected = [next(r for r in records if r['precision']==precision and r['dimension']==3
                     and r['n_h']==4 and r['backend']==backend) for backend in backends]
    kind = 'fp32' if precision == 'float32' else 'fp64'
    bars = ax.bar(labels, [r[f'{kind}_flops']/1e9 for r in selected], color='#2878b5' if kind=='fp32' else '#d35e22')
    for bar, result in zip(bars, selected):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{bar.get_height():.1f}", ha='center', va='bottom')
    ax.set_title(precision.upper())
    ax.set_ylabel('Billion executed add/mul/FMA FLOPs per call')
    ax.tick_params(axis='x', rotation=20)
    ax.set_ylim(0, 82)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle('3D, 1M particles, n_h = 4: different implementations execute different amounts of arithmetic')
fig.tight_layout()
fig.savefig(archive/'executed-work-3d.png', dpi=170)
plt.close(fig)
for r in records:
    if r['dimension'] == 3 and r['n_h'] == 4:
        print(r['precision'], r['backend'], 'ms',round(r['event_ms'],3), 'TF/s',round(r['tflops_event'],3), 'peak%',round(r['nominal_peak_pct'],1))
