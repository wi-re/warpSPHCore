"""Legacy renormalization policies and dense inference against readable Warp."""
import torch
from warpSPHCore.type_config import get_torch_precision
from warpSPHCore.linalg import pinvSymmetric as warp_inverse
from warpSPHCore.pinv import pinv_warp as warp_policy
from warpSPHCore.pinv.native import pinvSymmetric as native_inverse, pinv_warp as native_policy


def test_dense_and_renormalization_parity(device):
    import warp as wp
    wp.init()
    dtype = get_torch_precision()
    for dimension in (1,2,3):
        seed = torch.arange(11*dimension*dimension,device=device,dtype=dtype).reshape(11,dimension,dimension).sin()
        matrix = seed @ seed.mT + torch.eye(dimension,device=device,dtype=dtype)*.4
        matrix[0] = 0
        matrix[1] = -torch.eye(dimension,device=device,dtype=dtype)
        if dimension > 1:
            matrix[2] = torch.diag(matrix.new_tensor([2.]+[1e-7]*(dimension-1)))
        counts = torch.tensor([1,7,8,9,2,6,5,4,9,12,0],device=device,dtype=torch.int32)
        tolerance = 3e-5 if dtype == torch.float32 else 2e-11
        torch.testing.assert_close(native_inverse(matrix),warp_inverse(matrix),rtol=tolerance,atol=tolerance)
        native = native_policy(matrix,counts,symmetric=True)
        reference = warp_policy(matrix,counts,symmetric=True)
        for actual,expected in zip(native,reference):
            torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
        if dimension == 2:
            cotangent = matrix.cos()
            left = matrix.clone().requires_grad_(); right = matrix.clone().requires_grad_()
            left_bar, = torch.autograd.grad((native_policy(left,counts)[0]*cotangent).sum(),left)
            right_bar, = torch.autograd.grad((warp_policy(right,counts)[0]*cotangent).sum(),right)
            torch.testing.assert_close(left_bar,right_bar,rtol=tolerance,atol=tolerance)
