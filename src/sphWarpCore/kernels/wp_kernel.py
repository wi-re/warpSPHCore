from typing import Any
import warp as wp
import numpy as np
from .utils import *
from ..mathutil import computeDistanceVec

from .kernelFunctions.wendland2 import *
from .kernelFunctions.wendland4 import *
from .kernelFunctions.wendland6 import *
from .kernelFunctions.cubicSpline import *
from .kernelFunctions.quarticSpline import *
from .kernelFunctions.quinticSpline import *
from .kernelFunctions.B7 import *
from .kernelFunctions.poly6 import *
from .kernelFunctions.spiky import *
from .kernelFunctions.viscosityKernel import *
from .kernelFunctions.adhesionKernel import *
from .kernelFunctions.cohesionKernel import *

# Supported Kernels:
# wendland2 [index = 0]
# wendland4 [index = 1]
# wendland6 [index = 2]
# cubicSpline [index = 10]
# quarticSpline [index = 11]
# quinticSpline [index = 12]
# B7 [index = 13]
# poly6 [index = 20]
# spiky [index = 21]
# viscosity [index = 30]
# adhesion [index = 31]
# cohesion [index = 32]


@wp.func
def eval_k(q: wp.float32, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_k(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_k(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_k(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_k(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_k(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_k(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_k(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):
        return cohesionKernel_k(q, dim)
    return np.nan

@wp.func
def eval_dkdq(q: wp.float32, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):
        return cohesionKernel_dkdq(q, dim)
    return np.nan

@wp.func
def eval_d2kdq2(q: wp.float32, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_d2kdq2(q, dim)
    return np.nan

@wp.func
def eval_d3kdq3(q: wp.float32, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_d3kdq3(q, dim)
    return np.nan

@wp.func
def eval_C_d(dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_C_d(dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_C_d(dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_C_d(dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_C_d(dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_C_d(dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_C_d(dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_C_d(dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_C_d(dim)
    return np.nan

@wp.func
def eval_kernelScale(kernel: wp.int32, dim: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_kernelScale(dim)
    return np.nan

@wp.func
def eval_packing(kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_packingRatio()
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_packingRatio()
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_packingRatio()
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_packingRatio()
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_packingRatio()
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_packingRatio()
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_packingRatio()
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_packingRatio()
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_packingRatio()
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_packingRatio()
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_packingRatio()
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_packingRatio()
    return np.nan


from .adjoints import *

# Actual Kernel Functionality

# Torch Versions
# def Kernel_Scale(kernel: KernelType, dim: int = 2):
#     return eval_kernelScale(kernel, dim)
# def Kernel_C_d(kernel: KernelType, dim: int = 2):
#     return eval_C_d(kernel, dim)
# def Kernel_N_H(kernel: KernelType, dim: int = 2):
#     packingRatio = eval_packingRatio(kernel)
#     fac = 2.0 if dim == 1 else (np.pi if dim == 2 else 4 * np.pi / 3)
#     N = fac * packingRatio**dim * eval_kernelScale(kernel, dim)**dim
#     return N
# def Kernel_xi(kernel: KernelType, dim: int = 2):
#     return eval_packingRatio(kernel) * eval_kernelScale(kernel, dim)
@wp.func
def sphKernelScale(kernel: wp.int32, dim: wp.int32):
    return eval_kernelScale(kernel, dim)
@wp.func
def sphKernelC_d(kernel: wp.int32, dim: wp.int32):
    return eval_C_d(kernel, dim)
@wp.func
def sphKernelN_H(kernel: wp.int32, dim: wp.int32):
    packingRatio = eval_packing(kernel)
    fac = 2.0 if dim == 1 else (np.pi if dim == 2 else 4 * np.pi / 3)
    N = fac * packingRatio**dim * eval_kernelScale(kernel, dim)**dim
    return N
@wp.func
def sphKernel_xi(kernel: wp.int32, dim: wp.int32):
    return eval_packing(kernel) * eval_kernelScale(kernel, dim)

# Torch Version
# @torch.jit.script
# def Kernel(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
#     dim = x.shape[1]
#     # r = torch.linalg.norm(x, dim = -1)
#     r = vectorNorm(x)
#     # r = checkpoint(vectorNorm, x)
#     q = r / h
#     return eval_k(kernel, q, dim) * eval_C_d(kernel, dim) / h**float(dim)

@wp.func
def sphKernel_(x: vector(dtype=wp.float32, length=Any), h: wp.float32, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    # r = safe_sqrt(wp.dot(x,x))
    q = r / h
    if q > 1.0:
        return 0.0
    return eval_k(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim)

@wp.func
def sphKernel(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value): # KernelMeanSymmetric or SuperSymmetric
        return (sphKernel_(xij,hi,kernel) + sphKernel_(xij,hj,kernel))/2.0
    return sphKernel_(xij, hij, kernel)

@wp.func
def sphKernel_ij(
    xij: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),
):
    hij = computePairwiseSupport(hi, hj, mode)
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value) or mode == wp.static(SupportScheme.SuperSymmetric.value): # KernelMeanSymmetric or SuperSymmetric
        return (sphKernel_(xij,hi,kernel) + sphKernel_(xij,hj,kernel))/2.0
    return sphKernel_(xij, hij, kernel)


# Torch Version
# @torch.jit.script
# def Kernel_Gradient(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
#     dim = x.shape[1]
#     # r = torch.linalg.norm(x, dim = -1)

#     r = vectorNorm(x)
#     # r = checkpoint(vectorNorm, x)
#     q = r / h
#     # grad = vectorNormalize(x)
#     # grad = checkpoint(vectorNormalize, x)
#     grad = torch.nn.functional.normalize(x, dim = -1)
#     normalizationTerm = eval_C_d(kernel, dim) / h**(dim + 1)
#     kernelTerm = eval_dkdq(kernel, q, dim)
#     normalizedKernelTerm = kernelTerm * normalizationTerm
#     return grad * normalizedKernelTerm.view(-1,1)

@wp.func
def sphGradient_(x: vector(dtype=wp.float32, length=Any), h: wp.float32, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    if q > 1.0:
        return type(x)(0.0)
    grad = vectorNormalize_warp(input = x)
    normalizationTerm = eval_C_d(dim, kernel) / iPow(h, dim + 1)
    kernelTerm = eval_dkdq(q, dim, kernel)
    normalizedKernelTerm = kernelTerm * normalizationTerm
    return grad * normalizedKernelTerm



@wp.func
def sphKernelGradient(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value): # KernelMeanSymmetric
        return (sphGradient_(xij,hi,kernel) + sphGradient_(xij,hj,kernel))/2.0
    elif mode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphGradient_(xij,hi,kernel) - sphGradient_(-xij,hj,kernel))
    return sphGradient_(xij, hij, kernel)


@wp.func
def sphKernelGradient_ij(
    xij: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),
):
    hij = computePairwiseSupport(hi, hj, mode)
    if mode == wp.static(SupportScheme.KernelMeanSymmetric.value): # KernelMeanSymmetric
        return (sphGradient_(xij,hi,kernel) + sphGradient_(xij,hj,kernel))/2.0
    if mode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphGradient_(xij,hi,kernel) - sphGradient_(-xij,hj,kernel))/2.0
    return sphGradient_(xij, hij, kernel)
# Torch Version
# def Kernel_Derivative(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
#     dim = x.shape[1]
#     r = vectorNorm(x)
#     q = r / h
#     # grad = vectorNormalize(x)
#     return eval_dkdq(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 1)

@wp.func
def sphKernelDerivative_(x: vector(dtype=wp.float32, length=Any), h: wp.float32, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    if q > 1.0:
        return 0.0
    return eval_dkdq(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 1)

@wp.func
def sphKernelDerivative(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),   
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    if mode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphKernelDerivative_(xij,hi,kernel) + sphKernelDerivative_(xij,hj,kernel))/2.0
    return sphKernelDerivative_(xij, hij, kernel)
    

# Torch Version
# def Kernel_Hessian(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
#     dim = x.shape[1]
#     r = vectorNorm(x)
#     q = r / h
#     eps = get_epsilon(x.dtype)

#     k1 = eval_dkdq(kernel, q, dim)   * eval_C_d(kernel, dim) / h**(dim + 1)
#     k2 = eval_d2kdq2(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 2)
#     s = (r**2 + eps**2 * h**2)

#     factorA = torch.einsum('nu, nv -> nuv', x, x) / s.view(-1,1,1)# + torch.where(q < 1e-5, )
#     for i in range(dim):
#         factorA[:,i,i] = torch.where(q < 1e-5, 1, factorA[:,i,i])
#     # factorA[:,0,0] = torch.where(q < 1e-5, 1, factorA[:,0,0])
#     # factorA[:,1,1] = torch.where(q < 1e-5, 1, factorA[:,1,1])

#     # factorA

#     factorB = -torch.einsum('nu, nv -> nuv', x, x) / (r**3 + eps**3 * h**3).view(-1,1,1)
#     factorB += torch.eye(dim, device = x.device, dtype = x.dtype) / (r + eps**2 * h).view(-1,1,1)

#     hessian = factorA * k2.view(-1,1,1) + factorB * k1.view(-1,1,1)
#     return hessian

from .adjoints import warp_eye

@wp.func
def sphKernelHessian_(x: vector(dtype=wp.float32, length=Any), h: wp.float32, kernel: wp.int32):
    r = vectorNorm_warp(x)
    dim = wp.int32(x.length)
    q = r / h
    eps = 1e-5
    
    k1 = eval_dkdq(q, dim, kernel)   * eval_C_d(dim, kernel) / iPow(h, dim + 1)
    k2 = eval_d2kdq2(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 2)
    s = (iPow(r, 2) + iPow(eps,2) *iPow(h,2))
    
    factorA = wp.outer(x, x) / s
    if q < eps:
        for i in range(dim):
            factorA[i,i] = 1
    
    factorB = - wp.outer(x,x) /  (iPow(r, 3) + iPow(eps, 3) *iPow(h, 3))
    factorB += warp_eye(x) / (r + iPow(eps, 2) * h)
    
    hessian = factorA * k2 + factorB * k1
    
@wp.func
def sphKernelHessian(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    if mode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphKernelHessian_(xij,hi,kernel) + sphKernelHessian_(xij,hj,kernel))/2.0
    return sphKernelHessian_(xij, hij, kernel)
    


# Torch Version
# def Kernel_Laplacian(kernel: KernelType, x: torch.Tensor, h:torch.Tensor):
#     dim = x.shape[1]
#     r = torch.linalg.norm(x, dim = -1)
#     q = r / h
#     eps = get_epsilon(x.dtype)
#     r_eps = r + eps * h

#     k1 = eval_dkdq(kernel, q, dim)   * eval_C_d(kernel, dim) / h**(dim + 1)
#     k2 = eval_d2kdq2(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 2)

#     s = (x**2).sum(dim=-1) / (r_eps**2)
#     s = torch.where(q < 1e-5, 1, s)
#     t = - (x**2).sum(dim=-1) / (r_eps**3)
#     t += dim / (r_eps)

#     laplacian = (s * k2 + t * k1)
#     laplacian[q < 1e-5] = 0
#     return laplacian

@wp.func 
def sphKernelLaplacian_(x: vector(dtype=wp.float32, length=Any), h: wp.float32, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r / h
    eps = 1e-5
    r_eps = r + eps * h
    
    k1 = eval_dkdq(q, dim, kernel)   * eval_C_d(dim, kernel) / iPow(h, dim + 1)
    k2 = eval_d2kdq2(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 2)
    
    s = wp.dot(x,x) / iPow(r_eps, 2)
    if q < eps:
        s = wp.float32(1.0)
    t = - wp.dot(x,x) / iPow(r_eps, 3)
    t += wp.float32(dim) / r_eps
    
    laplacian = s * k2 + t * k1
    if q < eps or q > 1.0:
        laplacian = wp.float32(0.0)
    return laplacian

@wp.func
def sphKernelLaplacian(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    if mode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphKernelLaplacian_(xij,hi,kernel) + sphKernelLaplacian_(xij,hj,kernel))/2.0
    
    return sphKernelLaplacian_(xij, hij, kernel)


# Torch Version
# def Kernel_DkDh(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
#     dim = x.shape[1]
#     r = torch.linalg.norm(x, dim = -1)
#     q = r / h

#     k = eval_k(kernel, q, dim)
#     dkdq = eval_dkdq(kernel, q, dim)

#     normConstant = -float(Kernel_C_d(kernel, dim))/ h ** float(dim + 2)

#     return normConstant * (float(dim) * h * k + r * dkdq)

@wp.func
def sphKernelDkDh_(x: vector(dtype=wp.float32, length=Any), h: wp.float32, kernel: wp.int32):
    dim = wp.int32(x.length)
    r = vectorNorm_warp(x)
    q = r/h
    
    k = eval_k(q, dim, kernel)
    dkdq = eval_dkdq(q, dim, kernel)
    
    normConstant = - eval_C_d(dim, kernel) / iPow(h, dim + 2)
    
    return normConstant * (wp.float32(dim) * h * k + r * dkdq)

@wp.func
def sphKernelDkDh(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodic: wp.array(dtype = wp.bool),
    minDomain: wp.array(dtype = wp.float32),
    maxDomain: wp.array(dtype = wp.float32),
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = computeDistanceVec(xi, xj, periodic, minDomain, maxDomain)
    if mode == wp.static(SupportScheme.SuperSymmetric.value): # SuperSymmetric
        return (sphKernelDkDh_(xij,hi,kernel) + sphKernelDkDh_(xij,hj,kernel))/2.0
    
    return sphKernelDkDh_(xij, hij, kernel)
    

@wp.func
def get_dim(v: vector(length=1, dtype = wp.float32)): # type: ignore
    return 1
@wp.func
def get_dim(v: vector(length=2, dtype = wp.float32)): # type: ignore
    return 2
@wp.func
def get_dim(v: vector(length=3, dtype = wp.float32)): # type: ignore
    return 3

@wp.func
def get_dim(v: wp.array(dtype = vector(length=1, dtype = wp.float32))): # type: ignore
    return 1
@wp.func
def get_dim(v: wp.array(dtype = vector(length=2, dtype = wp.float32))): # type: ignore
    return 2
@wp.func
def get_dim(v: wp.array(dtype = vector(length=3, dtype = wp.float32))): # type: ignore
    return 3

@wp.func
def get_dim(v: matrix(shape=(1,1), dtype = wp.float32)): # type: ignore
    return 1
@wp.func
def get_dim(v: matrix(shape=(2,2), dtype = wp.float32)): # type: ignore
    return 2    
@wp.func
def get_dim(v: matrix(shape=(3,3), dtype = wp.float32)): # type: ignore
    return 3
@wp.func
def get_dim(v: wp.array(dtype = matrix(shape=(1,1), dtype = wp.float32))): # type: ignore
    return 1    
@wp.func
def get_dim(v: wp.array(dtype = matrix(shape=(2,2), dtype = wp.float32))): # type: ignore
    return 2
@wp.func
def get_dim(v: wp.array(dtype = matrix(shape=(3,3), dtype = wp.float32))): # type: ignore
    return 3

@wp.func
def correctGradientCRK(
    W_ij: wp.float32,
    gradW_ij: vector(length=Any, dtype=wp.float32), # type: ignore
    x_ij: vector(length=Any, dtype=wp.float32), # type: ignore
    Ai: wp.float32, Bi: vector(length=Any, dtype=wp.float32), gradAi: vector(length=Any, dtype=wp.float32), gradBi: matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
    dim: wp.int32
):
    
    term1 = (Ai * W_ij) * Bi 
    term2 = (Ai * (1.0 + wp.dot(Bi, x_ij))) * gradW_ij
    term3 = ((1.0 + wp.dot(Bi, x_ij)) * W_ij) * gradAi

    factor = Ai * W_ij
    # Compute the dot product of x_ij with each row of gradBi
    product = type(gradW_ij)(0.0)
    for row in range(dim):
        for col in range(dim):
            product[row] += x_ij[col] * gradBi[row, col]
    term4 = factor * product

    return term1 + term2 + term3  + term4


@wp.func
def computeKernelGradientCRK(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodicity: wp.array(dtype = wp.bool),
    domainMin: wp.array(dtype = wp.float32),
    domainMax: wp.array(dtype = wp.float32),
    useCRK: wp.bool,
    Ai: wp.float32, Bi: vector(length=Any, dtype=wp.float32), gradAi: vector(length=Any, dtype=wp.float32), gradBi: matrix(shape=(Any, Any), dtype=wp.float32), # type: ignore
):
    x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
    kernelGradient = sphKernelGradient_ij(x_ij, hi, hj, kernel, mode, periodicity, domainMin, domainMax)

    if useCRK:
        W_ij = sphKernel_ij(x_ij, hi, hj, kernel, mode, periodicity, domainMin, domainMax)
        return correctGradientCRK(
            W_ij,
            kernelGradient, 
            x_ij, 
            Ai, Bi, gradAi, gradBi, 
            get_dim(xi))
    return kernelGradient

@wp.func
def computeKernelCRK(
    xi: vector(dtype=wp.float32, length=Any),
    xj: vector(dtype=wp.float32, length=Any),
    hi: wp.float32,
    hj: wp.float32,
    kernel: wp.int32,
    mode: wp.uint32,
    periodicity: wp.array(dtype = wp.bool),
    domainMin: wp.array(dtype = wp.float32),
    domainMax: wp.array(dtype = wp.float32),
    useCRK: wp.bool,
    Ai: wp.float32, Bi: vector(length=Any, dtype=wp.float32)
):
    x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
    w_ij = sphKernel_ij(x_ij, hi, hj, kernel, mode, periodicity, domainMin, domainMax)
    if useCRK:
        xij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
        return Ai * (1.0 + wp.dot(Bi, xij)) * w_ij
    return w_ij
