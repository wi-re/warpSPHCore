from typing import Any
import warp as wp
import numpy as np
from .utils import *

from .kernelFunctions.wendland2 import *
from .kernelFunctions.wendland4 import *
from .kernelFunctions.wendland6 import *
from .kernelFunctions.cubicSpline import *
from .kernelFunctions.quarticSpline import *
from .kernelFunctions.quinticSpline import *
from .kernelFunctions.B7 import *
from .kernelFunctions.poly6 import *
from .kernelFunctions.spiky import *
from .kernelFunctions.viscosity import *
from .kernelFunctions.adhesion import *
from .kernelFunctions.cohesion import *

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
    if kernel == 0:
        return wendland2_k(q, dim)
    elif kernel == 1:
        return wendland4_k(q, dim)
    elif kernel == 2:
        return wendland6_k(q, dim)
    elif kernel == 10:
        return cubicSpline_k(q, dim)
    elif kernel == 11:
        return quarticSpline_k(q, dim)
    elif kernel == 12:
        return quinticSpline_k(q, dim)
    elif kernel == 13:
        return B7_k(q, dim)
    elif kernel == 20:
        return poly6_k(q, dim)
    elif kernel == 21:
        return spiky_k(q, dim)
    elif kernel == 30:
        return viscosity_k(q, dim)
    elif kernel == 31:
        return adhesion_k(q, dim)
    elif kernel == 32:         
        return cohesion_k(q, dim)
    return np.nan

@wp.func
def eval_dkdq(q: wp.float32, dim: wp.int32, kernel: wp.int32):
    if kernel == 0:
        return wendland2_dkdq(q, dim)
    elif kernel == 1:
        return wendland4_dkdq(q, dim)
    elif kernel == 2:
        return wendland6_dkdq(q, dim)
    elif kernel == 10:
        return cubicSpline_dkdq(q, dim)
    elif kernel == 11:
        return quarticSpline_dkdq(q, dim)
    elif kernel == 12:
        return quinticSpline_dkdq(q, dim)
    elif kernel == 13:
        return B7_dkdq(q, dim)
    elif kernel == 20:
        return poly6_dkdq(q, dim)
    elif kernel == 21:
        return spiky_dkdq(q, dim)
    elif kernel == 30:
        return viscosity_dkdq(q, dim)
    elif kernel == 31:
        return adhesion_dkdq(q, dim)
    elif kernel == 32:         
        return cohesion_dkdq(q, dim)
    return np.nan

@wp.func
def eval_d2kdq2(q: wp.float32, dim: wp.int32, kernel: wp.int32):
    if kernel == 0:
        return wendland2_d2kdq2(q, dim)
    elif kernel == 1:
        return wendland4_d2kd2q(q, dim)
    elif kernel == 2:
        return wendland6_d2kd2q(q, dim)
    elif kernel == 10:
        return cubicSpline_d2kd2q(q, dim)
    elif kernel == 11:
        return quarticSpline_d2kd2q(q, dim)
    elif kernel == 12:
        return quinticSpline_d2kd2q(q, dim)
    elif kernel == 13:
        return B7_d2kd2q(q, dim)
    elif kernel == 20:
        return poly6_d2kd2q(q, dim)
    elif kernel == 21:
        return spiky_d2kd2q(q, dim)
    elif kernel == 30:
        return viscosity_d2kd2q(q, dim)
    elif kernel == 31:
        return adhesion_d2kd2q(q, dim)
    elif kernel == 32:         
        return cohesion_d2kd2q(q, dim)
    return np.nan

@wp.func
def eval_d3kdq3(q: wp.float32, dim: wp.int32, kernel: wp.int32):
    if kernel == 0:
        return wendland2_d3kdq3(q, dim)
    elif kernel == 1:
        return wendland4_d3kd3q(q, dim)
    elif kernel == 2:
        return wendland6_d3kd3q(q, dim)
    elif kernel == 10:
        return cubicSpline_d3kd3q(q, dim)
    elif kernel == 11:
        return quarticSpline_d3kd3q(q, dim)
    elif kernel == 12:
        return quinticSpline_d3kd3q(q, dim)
    elif kernel == 13:
        return B7_d3kd3q(q, dim)
    elif kernel == 20:
        return poly6_d3kd3q(q, dim)
    elif kernel == 21:
        return spiky_d3kd3q(q, dim)
    elif kernel == 30:
        return viscosity_d3kd3q(q, dim)
    elif kernel == 31:
        return adhesion_d3kd3q(q, dim)
    elif kernel == 32:         
        return cohesion_d3kd3q(q, dim)
    return np.nan

@wp.func
def eval_C_d(dim: wp.int32, kernel: wp.int32):
    if kernel == 0:
        return wendland2_C_d(dim)
    elif kernel == 1:
        return wendland4_C_d(dim)
    elif kernel == 2:
        return wendland6_C_d(dim)
    elif kernel == 10:
        return cubicSpline_C_d(dim)
    elif kernel == 11:
        return quarticSpline_C_d(dim)
    elif kernel == 12:
        return quinticSpline_C_d(dim)
    elif kernel == 13:
        return B7_C_d(dim)
    elif kernel == 20:
        return poly6_C_d(dim)
    elif kernel == 21:
        return spiky_C_d(dim)
    elif kernel == 30:
        return viscosity_C_d(dim)
    elif kernel == 31:
        return adhesion_C_d(dim)
    elif kernel == 32:         
        return cohesion_C_d(dim)
    return np.nan

@wp.func
def eval_kernelScale(dim: wp.int32, kernel: wp.int32):
    if kernel == 0:
        return wendland2_kernelScaleernelScale(dim)
    elif kernel == 1:
        return wendland4_kernelScale(dim)
    elif kernel == 2:
        return wendland6_kernelScale(dim)
    elif kernel == 10:
        return cubicSpline_kernelScale(dim)
    elif kernel == 11:
        return quarticSpline_kernelScale(dim)
    elif kernel == 12:
        return quinticSpline_kernelScale(dim)
    elif kernel == 13:
        return B7_kernelScale(dim)
    elif kernel == 20:
        return poly6_kernelScale(dim)
    elif kernel == 21:
        return spiky_kernelScale(dim)
    elif kernel == 30:
        return viscosity_kernelScale(dim)
    elif kernel == 31:
        return adhesion_kernelScale(dim)
    elif kernel == 32:         
        return cohesion_kernelScale(dim)
    return np.nan

@wp.func
def eval_packing(kernel: wp.int32):
    if kernel == 0:
        return wendland2_packingRatio()
    elif kernel == 1:
        return wendland4_packingRatio()
    elif kernel == 2:
        return wendland6_packingRatio()
    elif kernel == 10:
        return cubicSpline_packingRatio()
    elif kernel == 11:
        return quarticSpline_packingRatio()
    elif kernel == 12:
        return quinticSpline_packingRatio()
    elif kernel == 13:
        return B7_packingRatio()
    elif kernel == 20:
        return poly6_packingRatio()
    elif kernel == 21:
        return spiky_packingRatio()
    elif kernel == 30:
        return viscosity_packingRatio()
    elif kernel == 31:
        return adhesion_packingRatio()
    elif kernel == 32:         
        return cohesion_packingRatio()
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
    mode: wp.uint32
):
    hij = computePairwiseSupport(hi, hj, mode)
    xij = xi - xj
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

# Torch Version
# def Kernel_Derivative(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
#     dim = x.shape[1]
#     r = vectorNorm(x)
#     q = r / h
#     # grad = vectorNormalize(x)
#     return eval_dkdq(kernel, q, dim) * eval_C_d(kernel, dim) / h**(dim + 1)


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


# Torch Version
# def Kernel_DkDh(kernel: KernelType, x: torch.Tensor, h: torch.Tensor):
#     dim = x.shape[1]
#     r = torch.linalg.norm(x, dim = -1)
#     q = r / h

#     k = eval_k(kernel, q, dim)
#     dkdq = eval_dkdq(kernel, q, dim)

#     normConstant = -float(Kernel_C_d(kernel, dim))/ h ** float(dim + 2)

#     return normConstant * (float(dim) * h * k + r * dkdq)