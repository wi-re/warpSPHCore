"""Native grid keys preserve the reference's full layout and traversal order."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_native_grid_layout_and_order_without_warp(tmp_path, precision):
    snapshots = {}
    for runtime, operators in (('warp', 'warp'), ('warp', 'native'), ('native', 'native')):
        name = runtime + '-' + operators
        output = tmp_path / (name + '.pt')
        env = os.environ | dict(WARPSPHCORE_BACKEND=runtime,
            WARPSPHCORE_OPERATORS=operators, WARPSPHCORE_NATIVE_BUILD='cpu',
            warpSPHCore_PRECISION=precision, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('host_grid_probe.py')),
            '--output', str(output)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        snapshots[name] = torch.load(output, weights_only=True)
    for name in ('warp-native', 'native-native'):
        assert snapshots[name].keys() == snapshots['warp-warp'].keys()
        for key, value in snapshots[name].items():
            reference = snapshots['warp-warp'][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, reference, rtol=0, atol=0, msg=key)
            else:
                assert value == reference, key
