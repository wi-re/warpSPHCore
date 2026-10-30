"""Real separate-process Gloo tests; NCCL uses the same worker explicitly."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch.distributed as dist


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo required')
@pytest.mark.parametrize('ranks', [2, 3])
@pytest.mark.parametrize('checks',['exact','contract'])
def test_process_owned_counts_operators_lifecycle_and_errors(tmp_path, ranks,checks):
    worker = Path(__file__).with_name('_rowChunkWorker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={ranks}',
           str(worker),'--replica-checks',checks,'--device','cpu','--output',str(tmp_path)]
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


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('ranks,seed',[(4,991),(5,3371)])
@pytest.mark.parametrize('checks',['exact','contract'])
def test_seeded_partition_capacity_faults_and_pair_lifetimes(tmp_path,ranks,seed,checks):
    worker=Path(__file__).with_name('_rowChunkStressWorker.py')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={ranks}',
        str(worker),'--device','cpu','--cycles','32','--seed',str(seed),'--replica-checks',checks,'--output',str(tmp_path)]
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (tmp_path/'run.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    assert result.stdout.count('all stress checks passed')==ranks,result.stdout


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('rank',[0,1])
def test_fatal_sender_or_receiver_after_transfer_contract_terminates_peers(tmp_path,rank):
    worker=Path(__file__).with_name('_rowChunkStressWorker.py')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
        str(worker),'--die-transfer-rank',str(rank),'--output',str(tmp_path)]
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    (tmp_path/'run.log').write_text(result.stdout)
    assert result.returncode!=0
    assert 'injected fatal participant after transfer contract' in result.stdout,result.stdout
    assert 'all stress checks passed' not in result.stdout
