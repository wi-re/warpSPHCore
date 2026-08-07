from ..type_config import *
from ..math import *
import warp as wp
from ..kernels import *
from warp.types import vector, matrix


@wp.func
def correctGradientCRK(
    W_ij: scalar_t,
    gradW_ij: vector(length=dim_t, dtype=scalar_t), # type: ignore
    x_ij: vector(length=dim_t, dtype=scalar_t), # type: ignore
    Ai: scalar_t, Bi: vector(length=dim_t, dtype=scalar_t), gradAi: vector(length=dim_t, dtype=scalar_t), gradBi: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
    dim: wp.int32
):
    
    term1 = (Ai * W_ij) * Bi 
    term2 = (Ai * (scalar_t(1.0) + wp.dot(Bi, x_ij))) * gradW_ij
    term3 = ((scalar_t(1.0) + wp.dot(Bi, x_ij)) * W_ij) * gradAi

    factor = Ai * W_ij
    # Compute the dot product of x_ij with each row of gradBi
    product = type(gradW_ij)(scalar_t(0.0))
    for row in range(dim):
        for col in range(dim):
            product[row] += x_ij[col] * gradBi[row, col]
    term4 = factor * product

    return term1 + term2 + term3  + term4


@wp.func
def computeKernelGradientCRK(
    xi: vector(dtype=scalar_t, length=dim_t),
    xj: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernel: wp.int32,
    mode: wp.uint32,
    periodicity: wp.array(dtype = wp.bool),
    domainMin: wp.array(dtype = scalar_t),
    domainMax: wp.array(dtype = scalar_t),
    useCRK: wp.bool,
    Ai: scalar_t, Bi: vector(length=dim_t, dtype=scalar_t), gradAi: vector(length=dim_t, dtype=scalar_t), gradBi: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
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
    xi: vector(dtype=scalar_t, length=dim_t),
    xj: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernel: wp.int32,
    mode: wp.uint32,
    periodicity: wp.array(dtype = wp.bool),
    domainMin: wp.array(dtype = scalar_t),
    domainMax: wp.array(dtype = scalar_t),
    useCRK: wp.bool,
    Ai: scalar_t, Bi: vector(length=dim_t, dtype=scalar_t)
):
    x_ij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
    w_ij = sphKernel_ij(x_ij, hi, hj, kernel, mode, periodicity, domainMin, domainMax)
    if useCRK:
        xij = computeDistanceVec(xi, xj, periodicity, domainMin, domainMax)
        return Ai * (scalar_t(1.0) + wp.dot(Bi, xij)) * w_ij
    return w_ij
