"""Native-only tensor signatures/helpers remain unchanged by hybrid routing."""
import torch
from warpSPHCore._backend import BACKEND
from warpSPHCore.linalg import applyPinv,pinvSymmetric,loadPadded3x3,symmetricPinv3
from warpSPHCore.pinv import native


def test_native_only_retains_tensor_signatures_and_helpers(device):
    assert BACKEND=='native'
    assert (applyPinv,pinvSymmetric,loadPadded3x3,symmetricPinv3)==(native.applyPinv,native.pinvSymmetric,native.loadPadded3x3,native.symmetricPinv3)
    matrix=torch.eye(2,device=device,dtype=torch.float64)[None]
    values=torch.ones(1,3,2,device=device,dtype=matrix.dtype)
    torch.testing.assert_close(pinvSymmetric(matrix=matrix),matrix)
    torch.testing.assert_close(applyPinv(values=values,matrix=matrix),values)
    padded=loadPadded3x3(matrix.reshape(1,-1),0,2)
    torch.testing.assert_close(symmetricPinv3(padded,1e-5),padded)
