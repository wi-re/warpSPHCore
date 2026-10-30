"""Real separate-process Gloo tests; NCCL uses the same worker explicitly."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch.distributed as dist


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo required')
@pytest.mark.parametrize('ranks', [2, 3])
def test_process_owned_counts_operators_lifecycle_and_errors(tmp_path, ranks):
    worker = Path(__file__).with_name('_rowChunkWorker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={ranks}',
           str(worker), '--device', 'cpu', '--output', str(tmp_path)]
    completed = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=180)
    assert completed.returncode == 0, completed.stdout
    assert completed.stdout.count('all checks passed') == ranks, completed.stdout


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo required')
def test_fatal_rank_failure_terminates_peers(tmp_path):
    worker = Path(__file__).with_name('_rowChunkWorker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
           str(worker), '--device', 'cpu', '--die-rank', '1']
    completed = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=90)
    assert completed.returncode != 0
    assert 'injected fatal rank failure' in completed.stdout
    assert 'all checks passed' not in completed.stdout
