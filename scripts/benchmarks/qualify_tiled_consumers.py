#!/usr/bin/env python3
"""Serial core-CG and downstream wave checks for opt-in tiled operators.

Requires the installed sibling warpSPH for the wave cases. Changes no sibling
files. Solver cases use explicit global lanes with only a Laplacian callback;
wave cases change just curl/Laplacian defaults, preserving other operators.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(solvers=[], waves=[], comparisons=[], complete=False, scope=__doc__)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(command, path, env):
        child = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if child.returncode:
            raise RuntimeError(f'{command}:\n{child.stdout[-8000:]}')
        return json.loads(path.read_text())

    with tempfile.TemporaryDirectory(prefix='tiled-consumers-') as directory:
        directory = Path(directory)
        for neighbors in (55, 220):
            for lanes in (1, 32):
                output = directory / f'cg-{neighbors}-{lanes}.json'
                command = [sys.executable, str(Path(__file__).with_name('bench_krylov.py')),
                           '--engine', 'native-cached', '--workload', 'sph', '--neighbors', str(neighbors),
                           '--ns', '1024', '65536', '1048576', '--iterations', '128', '--rtol', '1e-5',
                           '--repeat', '9', '--warmup', '4', '--output', str(output)]
                env = dict(os.environ, WARPSPHCORE_NEIGHBOR_LANES=str(lanes), warpSPHCore_PRECISION='float32')
                data = run(command, output, env)
                report['solvers'].append(dict(lanes=lanes, requested_neighbors=neighbors, command=command, result=data))
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(f'CG degree request {neighbors}, lanes {lanes}: '
                      f'{[round(row["median_wall_ms"], 3) for row in data["records"]]} ms', flush=True)
        for nx in (64, 256):
            states = {}
            trajectories = {}
            for lanes in (1, 32):
                output = directory / f'wave-{nx}-{lanes}.json'
                statePath = directory / f'wave-{nx}-{lanes}.pt'
                command = [sys.executable, str(Path(__file__).with_name('bench_lane_consumer.py')),
                           '--case', 'waveEquation', '--nx', str(nx), '--trial-lanes', str(lanes),
                           '--output', str(output), '--state-output', str(statePath)]
                data = run(command, output, dict(os.environ))
                report['waves'].append(data)
                states[lanes] = torch.load(statePath, map_location='cpu', weights_only=True)
                trajectories[lanes] = data['trajectory']
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(f'Wave nx={nx}, trial lanes={lanes}: {data["median_step_device_interval_ms"]:.4f} ms/step', flush=True)
            if states[1].keys() != states[32].keys():
                raise RuntimeError('consumer state fields differ')
            comparisons = {}
            for name, expected in states[1].items():
                actual = states[32][name]
                scale = max(float(expected.abs().max()), 1.)
                torch.testing.assert_close(actual, expected, rtol=5e-4, atol=1e-5 * scale)
                comparisons[name] = dict(max_abs_difference=float((actual - expected).abs().max()),
                                         relative_max_difference=float((actual - expected).abs().max()) / scale)
            # Same numerical interval, not merely the same number of steps.
            if [row.get('t') for row in trajectories[1]] != [row.get('t') for row in trajectories[32]]:
                raise RuntimeError('consumer time grids differ')
            report['comparisons'].append(dict(nx=nx, fields=comparisons, time_grid_equal=True))
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
