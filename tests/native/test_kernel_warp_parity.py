"""Raw radial and metadata parity, including knots, cutoffs and singularities."""
import os
import pytest
import torch
if os.environ.get('WARPSPHCORE_NATIVE_PARITY') != '1':
    pytest.skip('opt-in Warp/native reference process', allow_module_level=True)
import warp as wp
import warpSPHCore as sph
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.eval_kernel import eval_k, eval_dkdq, eval_d2kdq2, eval_d3kdq3, eval_C_d, eval_kernelScale, eval_packing
from warpSPHCore.kernels.native import evaluateKernel

@wp.kernel
def reference(q: wp.array(dtype=scalar_t), dim: int, kernel: int, out: wp.array2d(dtype=scalar_t)):
    i=wp.tid()
    if q[i] <= scalar_t(1.):
        out[i,0]=eval_k(q[i],dim,kernel)
        out[i,1]=eval_dkdq(q[i],dim,kernel)
        out[i,2]=eval_d2kdq2(q[i],dim,kernel)
        out[i,3]=eval_d3kdq3(q[i],dim,kernel)
    out[i,4]=eval_C_d(dim,kernel)
    out[i,5]=eval_kernelScale(kernel,dim)
    out[i,6]=eval_packing(kernel)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('dim',[1,2,3])
def test_raw_kernels_match_warp(device,kernel,dim):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('requires GPU')
    wp.init()
    dtype=torch.float64 if os.environ.get('warpSPHCore_PRECISION')=='float64' else torch.float32
    points=sorted(set([0.,1e-6,.1,1/7,.2,.214108111463,.25,1/3,3/7,.5,.6,2/3,5/7,.75,.9,1.,1.01]))
    q=torch.tensor(points,device=device,dtype=dtype)
    output=torch.zeros(len(points),7,device=device,dtype=dtype)
    wp.launch(reference,len(points),inputs=[wp.from_torch(q),dim,kernel.value,wp.from_torch(output)],device=device)
    wp.synchronize_device(device)
    x=torch.zeros(len(points),dim,device=device,dtype=dtype);x[:,0]=q
    actual=evaluateKernel(x,torch.ones_like(q),kernel).radial
    tol=2e-4 if dtype==torch.float32 else 2e-11
    torch.testing.assert_close(actual,output,rtol=tol,atol=tol,equal_nan=True)
