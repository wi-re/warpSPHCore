"""Torch adapter kernel outputs against independent derivative formulas."""
import math
import pytest
import torch
import warpSPHCore as sph
from warpSPHCore.kernels.native import evaluateKernel
from _support.kernels import SPECS, shape

@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_radial_derivative_ladder(device, kernel, dim, dtype):
    # Stay away from branch boundaries for differential tests; those are tested
    # separately against explicit one-sided conventions.
    points = [.54, .63, .78, .93] if kernel == sph.KernelFunctions.AdhesionKernel else [.037, .163, .371, .539, .691, .883]
    q = torch.tensor(points, device=device, dtype=torch.float64, requires_grad=True)
    expected = [shape(q, kernel, dim)]
    for _ in range(3):
        expected.append(torch.autograd.grad(expected[-1].sum(), q, create_graph=True)[0])
    x = torch.zeros(len(points), dim, dtype=dtype, device=device)
    x[:, 0] = q.detach().to(dtype)
    actual = evaluateKernel(x, torch.ones(len(points), device=device, dtype=dtype), kernel).radial[:, :4]
    tol = 3e-4 if dtype == torch.float32 else 3e-11
    torch.testing.assert_close(actual.double(), torch.stack(expected, 1).detach(), rtol=tol, atol=tol)


@pytest.mark.parametrize('kernel', list(sph.KernelFunctions))
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_spatial_and_support_derivatives(device, kernel, dim):
    q = .71 if kernel == sph.KernelFunctions.AdhesionKernel else .137
    x = torch.full((dim,), q/math.sqrt(dim)*1.3, dtype=torch.float64, device=device, requires_grad=True)
    h = torch.tensor(1.3, dtype=x.dtype, device=device, requires_grad=True)
    result = evaluateKernel(x.detach()[None].contiguous(), h.detach()[None], kernel)
    C = result.radial[0, 4].item()
    def value(x, h): return C * shape(x.norm()/h, kernel, dim) / h**dim
    expected = value(x, h)
    gx, gh = torch.autograd.grad(expected, (x, h), create_graph=True)
    H = torch.stack([torch.autograd.grad(gx[i], x, retain_graph=True)[0] for i in range(dim)])
    mixed = torch.stack([torch.autograd.grad(gx[i], h, retain_graph=True)[0] for i in range(dim)])
    for actual, reference in [(result.value[0], expected), (result.gradient[0], gx),
                              (result.hessian[0], H), (result.d_h[0], gh),
                              (result.gradient_d_h[0], mixed), (result.laplacian[0], H.trace())]:
        torch.testing.assert_close(actual, reference.detach(), rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize('name', list(SPECS))
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_normalization_moments_and_metadata(device, name, dim):
    kernel = sph.KernelFunctions[name]
    q = torch.linspace(0, 1, 20001, dtype=torch.float64, device=device)
    x = torch.zeros(q.numel(), dim, dtype=q.dtype, device=device); x[:, 0] = q
    result = evaluateKernel(x, torch.ones_like(q), kernel)
    table = result.radial
    sphere = [2, 2*math.pi, 4*math.pi][dim-1]
    C = table[0, 4].item()
    assert abs(C*sphere*torch.trapezoid(table[:, 0]*q**(dim-1), q).item()-1) < 2e-7
    sigma2 = C*sphere/dim*torch.trapezoid(table[:, 0]*q**(dim+1), q).item()
    assert abs(table[0, 5].item()-1/(2*math.sqrt(sigma2))) < 2e-6
    spec = SPECS[name]
    expected_C = eval(spec['C'][dim], {'__builtins__': {}, 'pi': math.pi}) if dim in spec['C'] else eval(spec['C'][str(dim)], {'__builtins__': {}, 'pi': math.pi})
    assert math.isclose(C, expected_C, rel_tol=2e-12)
    assert math.isclose(table[0, 6].item(), spec['packing']['value'], rel_tol=2e-12)


def test_support_origin_empty_and_invalid_inputs(device):
    x = torch.tensor([[0.,0.],[1.,0.],[1.01,0.]], device=device, dtype=torch.float64)
    h = torch.ones(3, device=device, dtype=x.dtype)
    result = evaluateKernel(x, h)
    torch.testing.assert_close(result.gradient[0], torch.zeros_like(x[0]), rtol=0, atol=0)
    torch.testing.assert_close(result.hessian[0], torch.eye(2,device=device,dtype=x.dtype)*(-20*7/math.pi), rtol=1e-14, atol=1e-14)
    assert result.value[-1] == 0 and result.gradient[-1].count_nonzero() == 0
    assert evaluateKernel(x[:0],h[:0]).value.numel() == 0
    assert evaluateKernel(x,torch.zeros_like(h)).value.count_nonzero() == 0
    # Negative-support diagnostics are exercised in the explicit debug process.
    with pytest.raises(TypeError): evaluateKernel(x,h,1234)


@pytest.mark.parametrize('kernel',[k for k in sph.KernelFunctions if k.value in {0,1,2,20,30,31,32,33,34,36,41}])
@pytest.mark.parametrize('dim',[1,2,3])
def test_smooth_origin_and_tiny_displacements(device,kernel,dim):
    x=torch.zeros(3,dim,device=device,dtype=torch.float64);x[1,0]=1e-20;x[2,0]=1e-12
    result=evaluateKernel(x,torch.ones(3,device=device,dtype=x.dtype),kernel)
    curvature=result.radial[0,2]*result.radial[0,4]
    assert torch.isfinite(result.hessian).all()
    torch.testing.assert_close(result.hessian[0],torch.eye(dim,device=device,dtype=x.dtype)*curvature,rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(result.gradient[:,0],curvature*x[:,0]+.5*result.radial[0,3]*result.radial[0,4]*x[:,0].abs()*x[:,0],rtol=2e-9,atol=1e-30)
    torch.testing.assert_close(result.hessian[1],result.hessian[0],rtol=2e-12,atol=2e-12)
