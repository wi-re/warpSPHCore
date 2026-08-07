from ..type_config import *
import warp as wp
from warp.types import vector, matrix
from typing import Any

# Unified Laplacian kernel: same design as the unified Gradient/Divergence/Curl kernels
# (see warpier_core.md's "Working Prototype -> Production" section). Laplacian shares
# Gradient's correction-path machinery (CRK, grad-h, volume, renormalization) and reuses
# GradientScheme to pick how the neighbor difference q_ij is formed (all four variants
# collapse to a (fj - fi)-based difference here -- see the comment below), then combines
# q_ij with the kernel gradient via one of `computeDotLaplacian`/`computeLaplacianDot2`/
# a direct second-derivative kernel evaluation, selected by LaplacianScheme. Unlike
# Gradient/Divergence/Curl, `positiveDivergence` is a real (non-decorative) part of the
# canonical ABI here.


@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=1), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + scalar_t(1e-12) * h_ij)
    dotx = q_ij * n_ij2[0]
    fkq = dotx * kernelGradient[0]
    result = type(q_ij)(-scalar_t(2.0) * fkq)
    return result

@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=2), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + scalar_t(1e-12) * h_ij)
    dotx = q_ij * n_ij2[0]
    doty = q_ij * n_ij2[1]
    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        output[i] += dotx[i] * kernelGradient[0] + doty[i] * kernelGradient[1]
    return -scalar_t(2.0) * output

@wp.func
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=3), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    n_ij2 = n_ij / (r_ij + scalar_t(1e-12) * h_ij)
    dotx = q_ij * n_ij2[0]
    doty = q_ij * n_ij2[1]
    dotz = q_ij * n_ij2[2]
    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        output[i] += dotx[i] * kernelGradient[0] + doty[i] * kernelGradient[1] + dotz[i] * kernelGradient[2]
    return -scalar_t(2.0) * output

@wp.func
def computeLaplacianDot2(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=Any), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96)
    # in https://www.sciencedirect.com/science/article/pii/S0021999110006753
    r_eps = r_ij + scalar_t(1e-8) * h_ij
    F_ab = wp.dot(n_ij, kernelGradient) / r_eps # this is a scalar

    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        # q_ij has internal shape [..., dim]; compute the dot product across the trailing
        # dim of q_ij and n_ij, then multiply by n_ij again for each output component.
        d = i % dim                  # component within trailing dim
        b = i // dim                 # block index over leading dims
        base = b * dim               # start of this block in flattened storage

        proj = scalar_t(scalar_t(0.0))
        for k in range(dim):
            proj += q_ij[base + k] * n_ij[k]

        left = scalar_t(dim + 2) * proj * n_ij[d]
        output[i] += -left * F_ab

    for i in range(inputLength):
        rightTerm = - q_ij[i] * F_ab
        output[i] += -rightTerm

    return output
