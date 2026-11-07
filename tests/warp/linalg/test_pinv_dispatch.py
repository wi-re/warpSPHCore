"""Hybrid Warp runtime/native tensor routing preserves kernel-callable helpers."""
import pytest
import torch
import warp as wp
import warpSPHCore as sph
import warpSPHCore.linalg as linalg
from warpSPHCore._backend import OPERATORS
from warpSPHCore.linalg import applyPinv, pinvSymmetric, loadPadded3x3, symmetricPinv3
from warpSPHCore.linalg import wp_pinv
from warpSPHCore.type_config import scalar_t


@wp.kernel
def _kernel_helpers(matrix:wp.array2d(dtype=scalar_t),output:wp.array3d(dtype=scalar_t),dimension:wp.int32):
    i=wp.tid()
    inverse=symmetricPinv3(loadPadded3x3(matrix,i,dimension),scalar_t(1e-6))
    for row in range(dimension):
        for column in range(dimension):
            output[i,row,column]=inverse[row,column]


@pytest.fixture(scope='module',autouse=True)
def init_warp():
    wp.init()


def test_tensor_exports_follow_operators_but_helpers_stay_warp():
    expected='warpSPHCore.linalg.pinv' if OPERATORS=='native' else 'warpSPHCore.linalg.wp_pinv'
    assert applyPinv.__module__==pinvSymmetric.__module__==expected
    assert sph.applyPinv is linalg.applyPinv and sph.pinvSymmetric is linalg.pinvSymmetric
    assert loadPadded3x3 is wp_pinv.loadPadded3x3
    assert symmetricPinv3 is wp_pinv.symmetricPinv3


@pytest.mark.parametrize('dimension',[1,2,3])
def test_tensor_primal_reaches_selected_launcher_and_handles_strides(dimension,device,monkeypatch):
    dtype=sph.get_torch_precision();diagonal=torch.tensor([2.,.1,.05][:dimension],device=device,dtype=dtype)
    matrix=torch.stack((torch.diag(diagonal),-torch.diag(diagonal),torch.zeros(dimension,dimension,device=device,dtype=dtype))).repeat_interleave(2,0)[::2].mT
    values=torch.arange(6*dimension,device=device,dtype=dtype).reshape(3,2,dimension).cos().repeat_interleave(2,1)[:,::2]
    expected=torch.linalg.pinv(matrix.double(),rtol=.1,hermitian=True).to(dtype);calls=[]
    if OPERATORS=='native':
        from warpSPHCore.native import _build
        extension=_build.build()
        class Observed:
            def small_inverse(self,*args):
                calls.append(args[3]);return extension.small_inverse(*args)
        monkeypatch.setattr(_build,'build',lambda:Observed())
        def reject(*args,**kwargs):raise AssertionError('native tensor pseudoinverse reached Warp launch')
        monkeypatch.setattr(wp,'launch',reject)
    else:
        launch=wp.launch
        def observed(*args,**kwargs):calls.append(kwargs['stream']);return launch(*args,**kwargs)
        monkeypatch.setattr(wp,'launch',observed)
    tolerance=4e-5 if dtype==torch.float32 else 2e-11
    inverse=pinvSymmetric(M=matrix,rtol=.1);product=applyPinv(V=values,M=matrix,rtol=.1)
    torch.testing.assert_close(inverse,expected,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(product,values@expected,rtol=tolerance,atol=tolerance)
    assert inverse.is_contiguous() and len(calls)==2
    if OPERATORS=='native':assert calls==[0,2]
    assert pinvSymmetric(matrix[:0]).shape==matrix[:0].shape
    assert applyPinv(values[:0],matrix[:0]).shape==values[:0].shape


@pytest.mark.parametrize('dimension',[1,2,3])
def test_kernel_helpers_remain_callable(dimension,device):
    dtype=sph.get_torch_precision();matrix=torch.diag(torch.arange(1,dimension+1,device=device,dtype=dtype))[None]
    output=torch.empty_like(matrix)
    wp.launch(_kernel_helpers,dim=1,inputs=[wp.from_torch(matrix.reshape(1,-1)),wp.from_torch(output),dimension],device=wp.device_from_torch(matrix.device))
    torch.testing.assert_close(output,torch.linalg.inv(matrix),rtol=4e-5 if dtype==torch.float32 else 2e-11,atol=4e-5 if dtype==torch.float32 else 2e-11)


@pytest.mark.parametrize('apply',[False,True])
@pytest.mark.parametrize('dimension',[2,3])
def test_selected_nonzero_discarded_branch_reverse_forward_and_higher(apply,dimension,device):
    if sph.get_torch_precision()!=torch.float64:pytest.skip('FP64 numerical/higher derivative contract')
    matrix=torch.diag(torch.tensor([2.,.1] if dimension==2 else [3.,1.,.1],device=device,dtype=torch.float64))[None].requires_grad_()
    values=torch.arange(dimension*2,device=device,dtype=matrix.dtype).reshape(1,2,dimension).cos().requires_grad_()
    function=(lambda m,v:applyPinv(v,m,rtol=.1)) if apply else (lambda m:pinvSymmetric(m,rtol=.1))
    inputs=(matrix,values) if apply else (matrix,)
    assert torch.autograd.gradcheck(function,inputs,fast_mode=True,check_forward_ad=True)
    assert torch.autograd.gradgradcheck(function,inputs,fast_mode=True)


def test_active_ad_retains_shared_torch_reverse_and_forward(device,monkeypatch):
    from warpSPHCore.pinv.symmetric import selectedSymmetricInverse
    from warpSPHCore.native import _build
    dtype=sph.get_torch_precision();matrix=torch.diag(torch.tensor([2.,.1],device=device,dtype=dtype))[None].requires_grad_();values=torch.tensor([[[.3,-.7],[.8,.2]]],device=device,dtype=dtype).requires_grad_()
    def reject(*args,**kwargs):raise AssertionError('active spectral AD left its shared Torch rule')
    monkeypatch.setattr(wp,'launch',reject);monkeypatch.setattr(_build,'build',reject)
    result=applyPinv(values,matrix,rtol=.1);expected=values@selectedSymmetricInverse(matrix,.1);seed=torch.ones_like(result)
    tolerance=4e-5 if dtype==torch.float32 else 5e-10
    torch.testing.assert_close(result,expected,rtol=tolerance,atol=tolerance)
    gradient=torch.autograd.grad(result,(matrix,values),seed);reference=torch.autograd.grad(expected,(matrix,values),seed)
    for a,b in zip(gradient,reference):torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
    dm,dv=torch.ones_like(matrix)*.03,torch.ones_like(values)*.02
    with torch.autograd.forward_ad.dual_level():
        m=torch.autograd.forward_ad.make_dual(matrix,dm);v=torch.autograd.forward_ad.make_dual(values,dv)
        actual=torch.autograd.forward_ad.unpack_dual(applyPinv(v,m,rtol=.1)).tangent
        reference=torch.autograd.forward_ad.unpack_dual(v@selectedSymmetricInverse(m,.1)).tangent
        torch.testing.assert_close(actual,reference,rtol=tolerance,atol=tolerance)
