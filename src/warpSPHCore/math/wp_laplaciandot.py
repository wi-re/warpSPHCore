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
def computeDotLaplacian(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), dim: wp.int32, base: wp.int32 # type: ignore
    ):
    proj = scalar_t(scalar_t(0.0))
    for k in range(dim):
        proj += q_ij[base + k] * n_ij[k]
    return proj


@wp.func
def computeLaplacianDot2(
    q_ij: vector(dtype = scalar_t, length=Any), n_ij: vector(dtype = scalar_t, length=Any), kernelGradient: vector(dtype = scalar_t, length=Any), r_ij: scalar_t, h_ij: scalar_t, inputLength: wp.int32, dim: wp.int32 # type: ignore
):
    # DJ Price Smoothed particle hydrodynamics and magnetohydrodynamics page 778 (eq 96)
    # in https://www.sciencedirect.com/science/article/pii/S0021999110006753
    #
    # KNOWN BUG (reverse-mode only, forward value unaffected): this function's automatic
    # (wp.Tape) adjoint w.r.t. n_ij is wrong -- confirmed via finite differences and via
    # Tier-2's independently-derived forward-mode JVP (coreOperations/wp_laplacianJVP.py's
    # computeSPHLaplacianDotGeometryJVP), which agree with each other and with finite
    # differences but not with wp.Tape's gradient here. Root cause (minimal repro in
    # docs/lessons_learned.md): n_ij is read via two different dynamically-indexed
    # expressions in the same runtime loop iteration (n_ij[k] inside the `proj` reduction
    # below, n_ij[d] outside it) -- when dim is small enough that k and d alias the same
    # element, Warp's adjoint under-accumulates that element's gradient contribution
    # (confirmed to exactly half its correct value in a minimal dim=1 repro). This is a
    # Warp code-generation limitation, not something fixable by restructuring this
    # function's math alone (a single merged loop was tried and does not fix it). Not
    # fixed here -- out of scope for the Tier-2 JVP work that found it; downstream code
    # differentiating through LaplacianScheme.Dot via .backward() gets a wrong gradient
    # until this is addressed. Forward-mode consumers (Tier-2's own JVP) are unaffected,
    # since they never go through this function's automatic adjoint.
    r_eps = r_ij + scalar_t(1e-8) * h_ij
    F_ab = wp.dot(n_ij, kernelGradient) / r_eps # this is a scalar

    output = type(q_ij)(scalar_t(0.0))
    for i in range(inputLength):
        # q_ij has internal shape [..., dim]; compute the dot product across the trailing
        # dim of q_ij and n_ij, then multiply by n_ij again for each output component.
        d = i % dim                  # component within trailing dim
        b = i // dim                 # block index over leading dims
        base = b * dim               # start of this block in flattened storage
        proj = computeDotLaplacian(q_ij, n_ij, dim, base)

        left = scalar_t(dim + 2) * proj * n_ij[d]
        output[i] += -left * F_ab

    for i in range(inputLength):
        rightTerm = - q_ij[i] * F_ab
        output[i] += -rightTerm

    return output
