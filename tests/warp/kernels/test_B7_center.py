"""The exact central spline polynomial keeps tiny derivatives and curvature."""
from contextlib import nullcontext
import torch
import warp as wp
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.kernelFunctions.B7 import B7_dkdq,B7_d2kdq2


@wp.kernel
def _central(q:wp.array(dtype=scalar_t),first:wp.array(dtype=scalar_t),second:wp.array(dtype=scalar_t)):
    i=wp.tid();first[i]=B7_dkdq(q[i],2);second[i]=B7_d2kdq2(q[i],2)


def test_central_derivative_and_reverse_curvature(device):
    wp.init()
    dtype=wp.dtype_to_torch(scalar_t)
    points=torch.tensor([0.,1e-20,1e-12,.07],device=device,dtype=dtype)
    q=wp.from_torch(points,requires_grad=True)
    first=wp.zeros(len(points),device=str(device),dtype=scalar_t,requires_grad=True)
    second=wp.zeros_like(first)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if points.is_cuda else None
    with wp.ScopedStream(stream) if stream is not None else nullcontext():
        with wp.Tape() as tape:
            wp.launch(_central,dim=len(points),inputs=[q],outputs=[first,second],device=str(device))
        tape.backward(grads={first:wp.ones_like(first)})
    expected=points*(-1320/343+points.square()*(240/7-120*points.square()))
    curvature=-1320/343+points.square()*(720/7-600*points.square())
    tolerance=3e-5 if dtype==torch.float32 else 2e-12
    torch.testing.assert_close(wp.to_torch(first),expected,rtol=tolerance,atol=1e-30)
    torch.testing.assert_close(wp.to_torch(second),curvature,rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(wp.to_torch(q.grad),curvature,rtol=tolerance,atol=tolerance)
    assert wp.to_torch(first)[0]==0
