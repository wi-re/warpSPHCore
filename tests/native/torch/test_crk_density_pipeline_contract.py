"""The corrected density's reverse through the complete preparation chain."""
import pytest
import torch
import warpSPHCore as sph
from _support.crk_factors_contract import case


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_full_rank_density_preparation_actual_reverse_and_forward_gradcheck(device,kernel):
    dimension=1+list(sph.KernelFunctions).index(kernel)%3
    values,functions=case(device,dimension,kernel=kernel,consistency=True)
    outputs=functions('native')(*values)
    assert outputs[1].isfinite().all() and (outputs[1]>0).all()
    assert torch.autograd.gradcheck(functions('native'),values,fast_mode=True,check_forward_ad=True,atol=2e-4,rtol=3e-3)
