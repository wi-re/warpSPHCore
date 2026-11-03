"""Compare paired benchmark states, including each saved accepted timestep."""
import argparse
import json
from pathlib import Path

import torch


def compare(reference, actual, tolerance):
    if reference.keys() != actual.keys():
        raise AssertionError('Saved fields differ')
    errors = {}
    for name, expected in reference.items():
        value = actual[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise AssertionError(f'{name}: shape/dtype mismatch')
        if not value.is_floating_point():
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            continue
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise AssertionError(f'{name}: nonfinite state')
        absolute = float((value - expected).abs().max())
        scale = max(1., float(expected.abs().max()))
        errors[name] = dict(max_absolute=absolute, scaled_max=absolute / scale)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--pattern', default='correctness-*')
    parser.add_argument('--tolerance', type=float, default=2.e-4,
                        help='Maximum error divided by max(1, max absolute reference field)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for directory in sorted(args.results.glob(args.pattern + '-warp')):
        native = directory.with_name(directory.name.removesuffix('-warp') + '-native')
        if not (directory / 'result.json').exists() or not (native / 'result.json').exists():
            raise AssertionError(f'Missing paired result: {directory.name}')
        expected = torch.load(directory / 'states.pt', weights_only=True)
        actual = torch.load(native / 'states.pt', weights_only=True)
        if len(expected['snapshots']) != len(actual['snapshots']):
            raise AssertionError('Accepted trajectory lengths differ')
        errors = [compare(expected['initial'], actual['initial'], args.tolerance)]
        for a, b in zip(expected['snapshots'], actual['snapshots']):
            errors.append(compare(a, b, args.tolerance))
        errors.append(compare(expected['final'], actual['final'], args.tolerance))
        a = json.loads((directory / 'result.json').read_text())
        b = json.loads((native / 'result.json').read_text())
        clock_error = abs(a['final_time'] - b['final_time'])
        precision = expected['final']['positions'].dtype
        # Physical safety limits depend on the evolving state, so nearly equal
        # fields can produce slightly different clocks. A picosecond floor
        # avoids requiring bit-identical adaptive decisions in FP64.
        clock_tolerance = max(1.e-12, 64 * torch.finfo(precision).eps * abs(a['final_time']))
        report = dict(name=directory.name.removesuffix('-warp'),
                      accepted_steps=len(actual['snapshots']),
                      clock_error=clock_error,
                      clock_tolerance=clock_tolerance,
                      max_scaled_error=max(e['scaled_max'] for state in errors for e in state.values()),
                      field_errors=errors, warp_ms=a['median_step_ms'], native_ms=b['median_step_ms'],
                      speedup=a['median_step_ms'] / b['median_step_ms'])
        report['passed'] = report['max_scaled_error'] <= args.tolerance and clock_error <= clock_tolerance
        reports.append(report)
        print(f"{report['name']}: error={report['max_scaled_error']:.3g}, passed={report['passed']}")
    if not reports:
        raise AssertionError('No paired results found')
    args.out.write_text(json.dumps(reports, indent=2))
    if any(not report['passed'] for report in reports):
        raise SystemExit('Trajectory tolerance failed; details retained in the comparison JSON')


if __name__ == '__main__':
    main()
