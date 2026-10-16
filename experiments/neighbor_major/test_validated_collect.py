"""The count-validated builder retains exact rows and rejects small capacity."""
import sys
from pathlib import Path

import pytest
import torch
import warp as wp
sys.path.insert(0,str(Path(__file__).resolve().parent))
import validated_collect
import test_layout
from warpSPHCore import SupportScheme


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    wp.init()
    return torch.device(request.param)


@pytest.mark.parametrize('dim',[2,3])
@pytest.mark.parametrize('periodic',[False,True])
@pytest.mark.parametrize('mode',list(SupportScheme))
def test_rows_and_order(device,dim,periodic,mode):
    with validated_collect.collectors():test_layout.test_direct_search_matches_production_order(device,periodic,dim,mode)


def test_capacity_before_launch(device):
    with validated_collect.collectors():test_layout.test_overflow_and_all_empty_rows(device)


def test_replay():
    with validated_collect.collectors():test_layout.test_replay_with_changed_neighbor_ids()
