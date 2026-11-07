"""Warp/native full-rank density-factor coupling with separate loss seeds."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import get_torch_precision
from _support.crk_factors_contract import case


@pytest.fixture(scope='module',autouse=True)
def runtime():wp.init()


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
def test_full_rank_density_pipeline_scalar_and_weighted_reverse(device,kernel):
    dimension=1+list(sph.KernelFunctions).index(kernel)%3
    values,functions=case(device,dimension,get_torch_precision(),kernel,consistency=True)
    tolerance=9e-3 if values[0].dtype==torch.float32 else 3e-6
    for weighted in (False,True):
        actual,expected=functions('native')(*values),functions('warp')(*values)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance,equal_nan=True)
        seeds=tuple(torch.linspace(-.3,.7,x.numel(),device=device,dtype=x.dtype).reshape_as(x) if weighted else torch.ones_like(x) for x in actual)
        bars=torch.autograd.grad(actual,values,seeds,allow_unused=True);wanted=torch.autograd.grad(expected,values,seeds,allow_unused=True)
        for a,b,x in zip(bars,wanted,values):
            torch.testing.assert_close(torch.zeros_like(x) if a is None else a,torch.zeros_like(x) if b is None else b,rtol=tolerance,atol=tolerance,equal_nan=True)
