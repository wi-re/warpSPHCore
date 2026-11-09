"""The renormalization facade honors numerical selection in a Warp host."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.pinv import wrapper


@pytest.mark.parametrize('dimension',[1,2,3])
def test_renormalization_native_selection(dimension,device,monkeypatch):
    wp.init()
    dtype=sph.get_torch_precision()
    diagonal=torch.arange(1,dimension+1,device=device,dtype=dtype)
    matrix=torch.stack([torch.diag(diagonal),-torch.diag(diagonal),torch.zeros(dimension,dimension,device=device,dtype=dtype),torch.eye(dimension,device=device,dtype=dtype)*1e-8])
    matrix=matrix.repeat_interleave(2,0)[::2].mT
    counts=torch.tensor([12,3,8,4],device=device,dtype=torch.int32)
    monkeypatch.setattr(wrapper,'OPERATORS','warp')
    reference=wrapper.pinv_warp(matrix,counts)
    monkeypatch.setattr(wrapper,'OPERATORS','native')
    from warpSPHCore.pinv.native import nt_pinv
    original=nt_pinv.pinv_warp
    calls=[]
    def observe(*a,**k):
        calls.append(1);return original(*a,**k)
    monkeypatch.setattr(nt_pinv,'pinv_warp',observe)
    def reject(*a,**k):raise AssertionError('native renormalization launched Warp')
    monkeypatch.setattr(wp,'launch',reject)
    actual=wrapper.pinv_warp(matrix,counts)
    assert calls==[1]
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b,rtol=4e-5 if dtype==torch.float32 else 5e-10,atol=4e-5 if dtype==torch.float32 else 5e-10)
