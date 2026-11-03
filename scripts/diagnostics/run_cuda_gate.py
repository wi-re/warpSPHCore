#!/usr/bin/env python3
"""Require a real CUDA device and run core gates in both process precisions.

Unlike CUDA-conditional steps on CPU runners, this exits unsuccessfully if no
GPU is available. Each precision uses a fresh interpreter because SPH kernel
types are selected at import. Includes native pinv/stream/capture, solver
contracts, neighbor lanes, graph rebuilds and host-readback checks.
"""

import os
from pathlib import Path
import subprocess
import sys

import torch
import warp as wp


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA gate requires a CUDA-capable Torch build and an accessible GPU')
    root = Path(__file__).resolve().parents[2]
    print(f'GPU: {torch.cuda.get_device_name()}; Torch: {torch.__version__}; Warp: {wp.__version__}', flush=True)
    for precision in ('float32', 'float64'):
        print(f'Process precision: {precision}', flush=True)
        env = dict(os.environ, warpSPHCore_PRECISION=precision)
        subprocess.run([sys.executable, '-m', 'pytest', '--suite', 'warp', '--device', 'cuda', 'tests/warp/linalg',
                        'tests/warp/radiusSearch', 'tests/warp/operations/test_neighbor_lanes.py',
                        'tests/warp/operations/test_neighbor_lane_selection.py',
                        'tests/warp/operations/test_tiled_curl_laplacian.py',
                        'tests/warp/operations/test_no_host_sync.py',
                        'tests/warp/operations/test_stream_ownership.py',
                        'tests/warp/operations/test_kind_directions.py',
                        'tests/warp/operations/test_capability_composition.py',
                        'tests/warp/operations/test_correction_numerics.py',
                        'tests/warp/operations/test_crk_reference_roles.py',
                        'tests/warp/operations/test_empty_operator_populations.py',
                        'tests/warp/operations/test_renorm_rank_jvp.py',
                        'tests/warp/operations/test_polyfit.py',
                        'tests/warp/operations/test_reconstruction.py',
                        'tests/warp/operations/test_reconstruction_contracts.py',
                        'tests/warp/operations/test_mfm.py',
                        'tests/warp/operations/test_mfm_warp.py',
                        'tests/warp/operations/test_mfm_contracts.py', '-q'],
                       cwd=root, env=env, check=True)
        subprocess.run([sys.executable, 'scripts/examples/run_all.py', '--device', 'cuda',
                        '--precision', precision], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
