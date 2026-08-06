import warp as wp
from ..type_config import scalar_t
from .wp_norm import norm_hess_warp, norm_warp, norm_grad_warp
from .wp_pow import iPow
from .wp_eps import get_epsilon
from warp.types import vector, matrix
from .wp_eye import warp_eye
from typing import Any
# from .wp_


@wp.func
def vectorNormalize_warp_1D(input: vector(dtype=scalar_t, length=1)):
    output = norm_grad_warp(input)
    return output
@wp.func_grad(vectorNormalize_warp_1D)
def adj_vectorNormalize_warp_1D(input: vector(dtype=scalar_t, length=1), adj_ret: vector(dtype=scalar_t, length=1)):
    tensor = norm_hess_warp(input)
    # there is no warp einsum, so we have to do this manually
    # output = wp.einsum('...ij, ...j -> ...i', tensor, adj_ret)
    output = wp.vector(scalar_t(0.0), length = input.length, dtype = input.dtype)
    for i in range(input.length):
        for j in range(input.length):
            output[i] += tensor[i][j] * adj_ret[j]
    wp.adjoint[input] += output

@wp.func
def vectorNormalize_warp_2D(input: vector(dtype=scalar_t, length=2)):
    output = norm_grad_warp(input)
    return output
@wp.func_grad(vectorNormalize_warp_2D)
def adj_vectorNormalize_warp_2D(input: vector(dtype=scalar_t, length=2), adj_ret: vector(dtype=scalar_t, length=2)):
    tensor = norm_hess_warp(input)
    # output = wp.einsum('...ij, ...j -> ...i', tensor, adj_ret)
    output = wp.vector(scalar_t(0.0), length = input.length, dtype = input.dtype)
    for i in range(input.length):
        for j in range(input.length):
            output[i] += tensor[i][j] * adj_ret[j]
    wp.adjoint[input] += output
    
@wp.func
def vectorNormalize_warp_3D(input: vector(dtype=scalar_t, length=3)):
    output = norm_grad_warp(input)
    return output
@wp.func_grad(vectorNormalize_warp_3D)
def adj_vectorNormalize_warp_3D(input: vector(dtype=scalar_t, length=3), adj_ret: vector(dtype=scalar_t, length=3)):
    tensor = norm_hess_warp(input)
    # output = wp.einsum('...ij, ...j -> ...i', tensor, adj_ret)
    output = wp.vector(scalar_t(0.0), length = input.length, dtype = input.dtype)
    for i in range(input.length):
        for j in range(input.length):
            output[i] += tensor[i][j] * adj_ret[j]
    wp.adjoint[input] += output
    
    
@wp.func
def vectorNormalize_warp(input: vector(dtype=scalar_t, length=1)):
    return vectorNormalize_warp_1D(input)
@wp.func
def vectorNormalize_warp(input: vector(dtype=scalar_t, length=2)):
    return vectorNormalize_warp_2D(input)
@wp.func
def vectorNormalize_warp(input: vector(dtype=scalar_t, length=3)):
    return vectorNormalize_warp_3D(input)



@wp.func
def vectorNorm_warp_1D(input: vector(dtype=scalar_t, length=1)):
    output = norm_warp(input)
    return output
@wp.func_grad(vectorNorm_warp_1D)
def adj_vectorNorm_warp(input: vector(dtype=scalar_t, length=1), adj_ret: scalar_t):
    normVector = vectorNormalize_warp(input)
    wp.adjoint[input] += normVector * adj_ret
    
@wp.func
def vectorNorm_warp_2D(input: vector(dtype=scalar_t, length=2)):
    output = norm_warp(input)
    return output
@wp.func_grad(vectorNorm_warp_2D)
def adj_vectorNorm_warp(input: vector(dtype=scalar_t, length=2), adj_ret: scalar_t):
    normVector = vectorNormalize_warp(input)
    wp.adjoint[input] += normVector * adj_ret
    
@wp.func
def vectorNorm_warp_3D(input: vector(dtype=scalar_t, length=3)):
    output = norm_warp(input)
    return output
@wp.func_grad(vectorNorm_warp_3D)
def adj_vectorNorm_warp(input: vector(dtype=scalar_t, length=3), adj_ret: scalar_t):
    normVector = vectorNormalize_warp(input)
    wp.adjoint[input] += normVector * adj_ret
    
@wp.func
def vectorNorm_warp(input: vector(dtype=scalar_t, length=1)):
    return vectorNorm_warp_1D(input)
@wp.func
def vectorNorm_warp(input: vector(dtype=scalar_t, length=2)):
    return vectorNorm_warp_2D(input)
@wp.func
def vectorNorm_warp(input: vector(dtype=scalar_t, length=3)):
    return vectorNorm_warp_3D(input)
    