#!/usr/bin/env python3
"""Export and associate separate Nsight captures with qualified inventory cases.

Requires matching bench_core_operators --profile-range JSON and Nsight report.
Checks Warp kernel order/names rather than guessing a case from launch size.
Retains raw CSV and all metric values/units. Never provides scaling samples.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

PREFIXES = {
    'density': 'computeSPHDensity_', 'interpolate': 'computeSPHInterpolation_',
    'gradient': 'computeSPHGradientTensor_', 'divergence': 'computeSPHDivergenceTensor_',
    'curl': 'computeSPHCurlTensor_', 'laplacian': 'computeSPHLaplacianTensor_',
    'covariance': 'computeCovariance_', 'crk_volume': 'computeCRKVolume_',
    'crk_moments': 'computeCRKMoments_', 'crk_density': 'computeCRKDensity_',
    'pinv_symmetric': '_pinvKernel_', 'apply_pinv': '_applyPinvKernel_',
    'pinv_policy': 'pinv2x2_warp_',
}


def associate(document, rows, units, source):
    """Fail closed if counter events do not match the recorded call sequence."""
    kernels = iter(row for row in rows if row['Kernel Name'].endswith('_cuda_kernel_forward'))
    records = []
    for case in document['records']:
        if case['status'] != 'profiled_without_timing':
            continue
        op = case['operation']
        expected = ['computeCovariance_', 'pinv2x2_warp_'] if op == 'renorm' and case['dim'] == 2 else [PREFIXES[op]]
        for prefix in expected:
            row = next(kernels, None)
            if row is None or not row['Kernel Name'].startswith(prefix):
                raise ValueError(f'counter/case order mismatch: {source}: {case}: expected {prefix}, got {row}')
            metrics = {}
            for name, value in row.items():
                try:
                    number = float(value.replace(',', ''))
                except (ValueError, AttributeError):
                    continue
                if not math.isfinite(number):
                    continue
                metrics[name] = dict(value=number, unit=units.get(name, ''))
            records.append(dict(case=case, source=source, kernel=row['Kernel Name'],
                                counter_id=row['ID'], block=row['Block Size'], grid=row['Grid Size'], metrics=metrics))
    if next(kernels, None) is not None:
        raise ValueError(f'unassigned Warp counter kernels: {source}')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path, nargs='+', help='completed --profile-range JSON files')
    p.add_argument('--ncu', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    report = dict(complete=False, scope=__doc__, inputs={}, records=[])
    for path in a.inputs:
        doc = json.loads(path.read_text())
        if not doc.get('complete'):
            raise ValueError(f'incomplete capture: {path}')
        captures = [path.with_suffix(ext) for ext in ['.ncu-repz', '.ncu-rep']]
        capture = next((f for f in captures if f.exists()), None)
        if capture is None:
            raise ValueError(f'missing Nsight report for {path}')
        output = path.with_suffix('.csv')
        with output.open('w') as stream:
            subprocess.run([str(a.ncu), '--import', str(capture), '--page', 'raw', '--csv'], stdout=stream, check=True)
        with output.open(newline='') as stream:
            reader = csv.DictReader(stream)
            units = next(reader)
            rows = list(reader)
        report['records'].extend(associate(doc, rows, units, str(path.resolve())))
        report['inputs'][str(path.resolve())] = dict(json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   report_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                                                   csv=str(output.resolve()), raw_cuda_kernels=len(rows),
                                                   statuses={status: sum(c['status'] == status for c in doc['records'])
                                                             for status in {c['status'] for c in doc['records']}})
    report['complete'] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(report["records"])} matched Warp kernel captures; raw auxiliary/vendor kernels remain in CSV')


if __name__ == '__main__':
    main()
