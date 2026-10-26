#!/usr/bin/env python3
"""Validate timing/work denominators and summarize saved core scaling sweeps.

CPU-only. Keeps source datasets distinct; no profiler durations are substituted
for synchronized public-call timing. Exposes scatter, large-N slopes and matched
CSR/grid or lane comparisons as evidence for the report.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(inputs):
    series = defaultdict(list)
    provenance = {}
    for path in inputs:
        document = json.loads(path.read_text())
        if not document.get('complete'):
            raise ValueError(f'incomplete dataset: {path}')
        provenance[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in document['records']:
            if row['status'] != 'qualified' or 'wall_ms' not in row:
                continue
            times = row['wall_ms']
            if len(times) < 2 or not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError('invalid timing samples')
            median = statistics.median(times)
            if not math.isclose(median, row['median_wall_ms'], rel_tol=1e-12):
                raise ValueError('stored median differs from samples')
            count = row['counts']['evaluated_nonself_pairs']
            if count is not None:
                expected = (row['edges'] - row['n']) * row['counts']['pair_passes']
                if count != expected or not math.isclose(count * 1000 / median, row['interactions_per_second'], rel_tol=1e-12):
                    raise ValueError('interaction count/rate mismatch')
            rate = row['interactions_per_second'] if count is not None else row['matrices_per_second']
            quartiles = statistics.quantiles(times, n=4, method='inclusive')
            metadata = dict(dataset=path.stem, dim=row['dim'], dtype=document['dtype'], operation=row['operation'],
                            neighbors=row['neighbors'], traversal=row['traversal'], lanes=row['lanes'], form=row['form'],
                            kernel=row['kernel'], support=row['support'], gradient=row['gradient'], laplacian=row['laplacian'],
                            field_shape=row.get('field_shape', []))
            point = dict(n=row['n'], median_wall_ms=median, median_device_interval_ms=statistics.median(row['device_interval_ms']),
                         iqr_fraction=(quartiles[2] - quartiles[0]) / median, min_ms=min(times), max_ms=max(times),
                         rate=rate, unit='interactions/s' if count is not None else 'matrices/s', degree=row['degree'],
                         peak_torch_allocated_bytes=row['peak_torch_allocated_bytes'])
            series[json.dumps(metadata, sort_keys=True)].append(point)
    output = []
    for key, points in series.items():
        points.sort(key=lambda p: p['n'])
        if len({p['n'] for p in points}) != len(points):
            raise ValueError('duplicate N in a measurement series')
        large = points[-3:]
        slopes = [math.log(b['median_wall_ms']/a['median_wall_ms']) / math.log(b['n']/a['n']) for a,b in zip(large,large[1:])]
        output.append(dict(metadata=json.loads(key), points=points, last_interval_log_slope=slopes[-1] if slopes else None,
                           max_iqr_fraction=max(p['iqr_fraction'] for p in points)))
    return dict(complete=True, source_inputs_sha256=provenance, series=output,
                case_count=sum(len(s['points']) for s in output),
                cases_iqr_over_10_percent=sum(p['iqr_fraction'] > .1 for s in output for p in s['points']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = summarize(a.inputs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{result["case_count"]} checked cases, {len(result["series"])} series; '
          f'{result["cases_iqr_over_10_percent"]} cases have IQR > 10% of median')


if __name__ == '__main__':
    main()
