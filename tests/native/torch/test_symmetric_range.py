"""Native primal range: finite eigenvalues must survive symmetrization."""
import pytest
import torch
import warpSPHCore as sph


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_large_finite_symmetric_inverse_and_application(device,dimension,dtype):
    scale=1e38
    normalized=torch.eye(dimension,dtype=dtype,device=device)
    normalized[0,1]=normalized[1,0]=.25
    if dimension==3:normalized[-1,-1]=-1
    matrix=(normalized*scale)[None]
    values=torch.arange(1,dimension+1,dtype=dtype,device=device)[None,None]*scale
    # Solve at identical stored inputs; the independent reference stays in FP64.
    expected=torch.linalg.pinv(matrix.double(),rtol=1e-6,hermitian=True)
    tolerance=4e-5 if dtype==torch.float32 else 5e-10
    inverse=sph.pinvSymmetric(matrix,rtol=1e-6)
    applied=sph.applyPinv(values,matrix,rtol=1e-6)
    assert torch.isfinite(inverse).all() and torch.isfinite(applied).all()
    torch.testing.assert_close(inverse*scale,(expected*scale).to(dtype),rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(applied,(values.double()@expected).to(dtype),rtol=tolerance,atol=tolerance)
    if device=='cpu':return
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured=sph.pinvSymmetric(matrix,rtol=1e-6)
        matrix.mul_(.5);graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(captured*scale,(expected*scale*2).to(dtype),rtol=tolerance,atol=tolerance)
