"""Expanded native primals against a dense, independently differentiated oracle."""
import pytest
import torch
import warpSPHCore as sph
if sph.backend != 'native': pytest.skip('native process only', allow_module_level=True)
from _support.operators import fixture, dense
pytestmark = pytest.mark.extended


OPS=[sph.WarpOperation.Interpolate,sph.WarpOperation.Gradient,sph.WarpOperation.Divergence,
     sph.WarpOperation.Curl,sph.WarpOperation.Laplacian,sph.WarpOperation.Covariance]

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
def test_primal_fields_and_schemes(device,op,dim,rank,scheme):
    if op==sph.WarpOperation.Divergence and rank==0:pytest.skip('requires vector field')
    if op==sph.WarpOperation.Curl and rank==0 and dim>1:pytest.skip('requires vector field')
    q,r,d,qf,rf=fixture(device,dim,rank)
    prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=op,gradientMode=scheme,
        supportMode=sph.SupportScheme.KernelMeanSymmetric,operationMode=sph.OperationDirection.AllToAll)
    actual=sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r)
    torch.testing.assert_close(actual,dense(q,r,d,prop,qf,rf),rtol=2e-10,atol=2e-10)


@pytest.mark.parametrize('kernel',list(sph.KernelFunctions))
@pytest.mark.parametrize('support',list(sph.SupportScheme)[:-1])
@pytest.mark.parametrize('op',[sph.WarpOperation.Density,sph.WarpOperation.Gradient,sph.WarpOperation.Interpolate])
def test_kernel_family_in_primals(device,kernel,support,op):
    q,r,d,qf,rf=fixture(device,2,0)
    prop=sph.OperationProperties(kernel,operation=op,gradientMode=sph.GradientScheme.Difference,
        supportMode=support,operationMode=sph.OperationDirection.TrueAllToToAll)
    actual=sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r)
    torch.testing.assert_close(actual,dense(q,r,d,prop,qf,rf),rtol=3e-9,atol=3e-9)


@pytest.mark.parametrize('mode',list(sph.LaplacianScheme))
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('scheme',list(sph.GradientScheme))
@pytest.mark.parametrize('rank',[0,1,2])
def test_laplacian_variants(device,mode,dim,scheme,rank):
    if rank==0 and mode==sph.LaplacianScheme.Dot and dim>1:pytest.skip('requires vector field')
    q,r,d,qf,rf=fixture(device,dim,rank)
    prop=sph.OperationProperties(sph.KernelFunctions.CubicSpline,operation=sph.WarpOperation.Laplacian,
        gradientMode=scheme,laplacianMode=mode,supportMode=sph.SupportScheme.SuperSymmetric,
        operationMode=sph.OperationDirection.FluidToBoundary)
    torch.testing.assert_close(sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r),dense(q,r,d,prop,qf,rf),rtol=2e-10,atol=2e-10)


def test_constant_laplacian_and_tensor_divergence_axes(device):
    q,r,d,qf,rf=fixture(device,3,2)
    for scheme in sph.GradientScheme:
        for mode in sph.LaplacianScheme:
            prop=sph.OperationProperties(sph.KernelFunctions.Wendland6,operation=sph.WarpOperation.Laplacian,
                gradientMode=scheme,laplacianMode=mode)
            out=sph.warpOperation(q,prop,d,torch.ones_like(qf),torch.ones_like(rf),referenceParticles=r)
            assert out.count_nonzero()==0
    for dot in (False,True):
        prop=sph.OperationProperties(sph.KernelFunctions.Wendland6,operation=sph.WarpOperation.Divergence,
            gradientMode=sph.GradientScheme.Difference,divergenceDotMode=dot)
        torch.testing.assert_close(sph.warpOperation(q,prop,d,qf,rf,referenceParticles=r),dense(q,r,d,prop,qf,rf),rtol=2e-10,atol=2e-10)
