from typing import Any
import warp as wp
import numpy as np
from .utils import *
from warp.types import vector, matrix


@wp.func
def safe_sqrt(x: float):
    return wp.sqrt(x)
@wp.func_grad(safe_sqrt)
def adj_safe_sqrt(x: float, adj_ret: float):
    if x > 0.0:
        wp.adjoint[x] += 1.0 / (2.0 * wp.sqrt(x)) * adj_ret

from warp.types import vector
# warp versions

@wp.func
def norm_warp(input: vector(dtype=wp.float32, length=Any)):
    return safe_sqrt(wp.dot(input, input))

@wp.func 
def norm_grad_warp(input: vector(dtype=wp.float32, length=Any)):
    length = norm_warp(input)
    float_eps = get_epsilon(length)
    length = wp.max(length, float_eps)
    return input / length


@wp.func
def warp_eye(input: vector(dtype=wp.float32, length=1)):
    retVal = matrix(shape = (1, 1), dtype=wp.float32)
    for i in range(1):
        for j in range(1):
            retVal[i][j] = 1.0 if i == j else 0.0
    return retVal
@wp.func
def warp_eye(input: vector(dtype=wp.float32, length=2)):
    retVal = matrix(shape = (2, 2), dtype=wp.float32)
    for i in range(2):
        for j in range(2):
            retVal[i][j] = 1.0 if i == j else 0.0
    return retVal
@wp.func
def warp_eye(input: vector(dtype=wp.float32, length=3)):
    retVal = matrix(shape = (3, 3), dtype=wp.float32)
    for i in range(3):
        for j in range(3):
            retVal[i][j] = 1.0 if i == j else 0.0
    return retVal

@wp.func
def norm_hess_warp(input: vector(dtype=wp.float32, length=Any)):
    eps = get_epsilon(input[0])
    r = norm_warp(input) + eps

    outerProd = wp.outer(input, input)
    diagTerm = warp_eye(input) * (iPow(r, 2) + iPow(eps, 3))

    tensor = 1.0/(iPow(r, 3) + iPow(eps, 3)) * (diagTerm - outerProd)

    return tensor

@wp.func
def vectorNormalize_warp_1D(input: vector(dtype=wp.float32, length=1)):
    output = norm_grad_warp(input)
    return output
@wp.func_grad(vectorNormalize_warp_1D)
def adj_vectorNormalize_warp_1D(input: vector(dtype=wp.float32, length=1), adj_ret: vector(dtype=wp.float32, length=1)):
    tensor = norm_hess_warp(input)
    # there is no warp einsum, so we have to do this manually
    # output = wp.einsum('...ij, ...j -> ...i', tensor, adj_ret)
    output = wp.vector(0.0, length = input.length, dtype = input.dtype)
    for i in range(input.length):
        for j in range(input.length):
            output[i] += tensor[i][j] * adj_ret[j]
    wp.adjoint[input] += output

@wp.func
def vectorNormalize_warp_2D(input: vector(dtype=wp.float32, length=2)):
    output = norm_grad_warp(input)
    return output
@wp.func_grad(vectorNormalize_warp_2D)
def adj_vectorNormalize_warp_2D(input: vector(dtype=wp.float32, length=2), adj_ret: vector(dtype=wp.float32, length=2)):
    tensor = norm_hess_warp(input)
    # output = wp.einsum('...ij, ...j -> ...i', tensor, adj_ret)
    output = wp.vector(0.0, length = input.length, dtype = input.dtype)
    for i in range(input.length):
        for j in range(input.length):
            output[i] += tensor[i][j] * adj_ret[j]
    wp.adjoint[input] += output
    
@wp.func
def vectorNormalize_warp_3D(input: vector(dtype=wp.float32, length=3)):
    output = norm_grad_warp(input)
    return output
@wp.func_grad(vectorNormalize_warp_3D)
def adj_vectorNormalize_warp_3D(input: vector(dtype=wp.float32, length=3), adj_ret: vector(dtype=wp.float32, length=3)):
    tensor = norm_hess_warp(input)
    # output = wp.einsum('...ij, ...j -> ...i', tensor, adj_ret)
    output = wp.vector(0.0, length = input.length, dtype = input.dtype)
    for i in range(input.length):
        for j in range(input.length):
            output[i] += tensor[i][j] * adj_ret[j]
    wp.adjoint[input] += output
    
    
# @wp.func
# def vectorNormalize_warp(input: vector(dtype=wp.float32, length=Any)):
#     dim = wp.int32(input.length)
#     if dim == 1:
#         return vectorNormalize_warp_1D(input)
#     elif dim == 2:
#         return vectorNormalize_warp_2D(input)
#     elif dim == 3:
#         return vectorNormalize_warp_3D(input)
#     else:
#         return vectorNormalize_warp_1D(input) * np.nan
@wp.func
def vectorNormalize_warp(input: vector(dtype=wp.float32, length=1)):
    return vectorNormalize_warp_1D(input)
@wp.func
def vectorNormalize_warp(input: vector(dtype=wp.float32, length=2)):
    return vectorNormalize_warp_2D(input)
@wp.func
def vectorNormalize_warp(input: vector(dtype=wp.float32, length=3)):
    return vectorNormalize_warp_3D(input)
    
    
# Torch version
# input, = ctx.saved_tensors
# # input = input.to(grad_output.device)
# grad_output = grad_output

# float_eps = get_epsilon(input.dtype)

# grad_input = torch.einsum('...i, ... -> ...i', vectorNormalize(input + float_eps*0), grad_output)
@wp.func
def vectorNorm_warp_1D(input: vector(dtype=wp.float32, length=1)):
    output = norm_warp(input)
    return output
@wp.func_grad(vectorNorm_warp_1D)
def adj_vectorNorm_warp(input: vector(dtype=wp.float32, length=1), adj_ret: wp.float32):
    normVector = vectorNormalize_warp(input)
    wp.adjoint[input] += normVector * adj_ret
    
@wp.func
def vectorNorm_warp_2D(input: vector(dtype=wp.float32, length=2)):
    output = norm_warp(input)
    return output
@wp.func_grad(vectorNorm_warp_2D)
def adj_vectorNorm_warp(input: vector(dtype=wp.float32, length=2), adj_ret: wp.float32):
    normVector = vectorNormalize_warp(input)
    wp.adjoint[input] += normVector * adj_ret
    
@wp.func
def vectorNorm_warp_3D(input: vector(dtype=wp.float32, length=3)):
    output = norm_warp(input)
    return output
@wp.func_grad(vectorNorm_warp_3D)
def adj_vectorNorm_warp(input: vector(dtype=wp.float32, length=3), adj_ret: wp.float32):
    normVector = vectorNormalize_warp(input)
    wp.adjoint[input] += normVector * adj_ret
    
# @wp.func
# def vectorNorm_warp(input: vector(dtype=wp.float32, length=Any)):
#     dim = wp.int32(input.length)
#     if dim == 1:
#         return vectorNorm_warp_1D(input)
#     elif dim == 2:
#         return vectorNorm_warp_2D(input)
#     elif dim == 3:
#         return vectorNorm_warp_3D(input)
#     else:
#         return vectorNorm_warp_1D(input) * np.nan
@wp.func
def vectorNorm_warp(input: vector(dtype=wp.float32, length=1)):
    return vectorNorm_warp_1D(input)
@wp.func
def vectorNorm_warp(input: vector(dtype=wp.float32, length=2)):
    return vectorNorm_warp_2D(input)
@wp.func
def vectorNorm_warp(input: vector(dtype=wp.float32, length=3)):
    return vectorNorm_warp_3D(input)
    