from ..type_config import *
from ..math import *
import warp as wp
from ..kernels import *
from ..kernels.kernelJVP import sphKernelGradientJVP
from warp.types import vector, matrix
from ..dataTypes import *

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
    # gradBi[g, c] = d(B[c]) / d(x_i[g]) -- DIFFERENTIATION index first,
    # component index second. That is the layout crk_terms.py's
    # computeCRKTermsWarp produces (its einsums index the output 'nki' with k
    # the gamma/derivative axis of dm_1dgamma/dm_2dgamma and i the component
    # axis of m_2_inv). The product-rule term is sum_c x_ij[c] dB[c]/dx[g]:
    # contract gradBi's SECOND (component) axis against x_ij, leaving the
    # differentiation index g free -- i.e. gradBi @ x_ij.
    #
    # Regression note (2026-09-26): 5ca1882 (2026-08-11) replaced the original
    # explicit loop `product[row] += x_ij[col] * gradBi[row, col]` (correct)
    # with matmul(wp.transpose(gradBi), x_ij) under the opposite layout
    # assumption. gradBi is not symmetric on disordered/boundary particles, so
    # that broke CRK's exact linear-gradient reproduction (flat ~1e-3 error in
    # the higher-order harness, higherOrderSPH/harness/FINDINGS.md section 1);
    # with gradBi @ x_ij it is exact to ~1e-15 again.
    product = matmul(gradBi, x_ij)
    term4 = factor * product

    return term1 + term2 + term3  + term4


@wp.func
def correctGradientCRKJVP(
    W_ij: scalar_t, dW_ij: scalar_t,
    gradW_ij: vector(length=dim_t, dtype=scalar_t), dgradW_ij: vector(length=dim_t, dtype=scalar_t), # type: ignore
    x_ij: vector(length=dim_t, dtype=scalar_t), dx_ij: vector(length=dim_t, dtype=scalar_t), # type: ignore
    Ai: scalar_t, Bi: vector(length=dim_t, dtype=scalar_t), gradAi: vector(length=dim_t, dtype=scalar_t), gradBi: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
    dAi: scalar_t, dBi: vector(length=dim_t, dtype=scalar_t), dgradAi: vector(length=dim_t, dtype=scalar_t), dgradBi: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
    dim: wp.int32
):
    """JVP of `correctGradientCRK` (`warpier_tier2_correction_jvp_plan.md`
    phase (c), Stage 4): ordinary product rule on that function's four-term
    formula, ported from `scripts/spikes/spike_forward_mode_tier2_crk.py`'s
    `assembled_correctedGradient_jvp`. `Ai`/`Bi`/`gradAi`/`gradBi` are the
    per-QUERY-particle CRK correction terms (constant across the neighbor
    loop, matching `correctGradientCRK`'s own convention -- see that
    function's callers, all of which pass `iCorrectionData.A/B/gradA/gradB`);
    `dAi`/`dBi`/`dgradAi`/`dgradBi` are their tangents.

    `term4`'s contraction needs the SAME axis care `correctGradientCRK`'s own
    comment spells out (`matmul(gradBi, x_ij)`: `gradBi` is stored
    [differentiation, component], so its second/component axis is contracted
    against `x_ij`, leaving the differentiation axis free) -- its tangent is
    an ordinary product rule over that same bilinear contraction, needing
    BOTH the `dgradBi` and `dx_ij` terms.
    """
    dot_Bx = wp.dot(Bi, x_ij)
    d_dot_Bx = wp.dot(dBi, x_ij) + wp.dot(Bi, dx_ij)

    term1 = (Ai * W_ij) * Bi
    dterm1 = (dAi * W_ij + Ai * dW_ij) * Bi + (Ai * W_ij) * dBi

    factor2 = Ai * (scalar_t(1.0) + dot_Bx)
    dfactor2 = dAi * (scalar_t(1.0) + dot_Bx) + Ai * d_dot_Bx
    term2 = factor2 * gradW_ij
    dterm2 = dfactor2 * gradW_ij + factor2 * dgradW_ij

    factor3 = (scalar_t(1.0) + dot_Bx) * W_ij
    dfactor3 = d_dot_Bx * W_ij + (scalar_t(1.0) + dot_Bx) * dW_ij
    term3 = factor3 * gradAi
    dterm3 = dfactor3 * gradAi + factor3 * dgradAi

    factor4 = Ai * W_ij
    dfactor4 = dAi * W_ij + Ai * dW_ij
    product = matmul(gradBi, x_ij)
    dproduct = matmul(dgradBi, x_ij) + matmul(gradBi, dx_ij)
    term4 = factor4 * product
    dterm4 = dfactor4 * product + factor4 * dproduct

    correctedG = term1 + term2 + term3 + term4
    dCorrectedG = dterm1 + dterm2 + dterm3 + dterm4
    return correctedG, dCorrectedG


@wp.func
def computeKernelGradientCRKJVP(
    xi: vector(dtype=scalar_t, length=dim_t), xj: vector(dtype=scalar_t, length=dim_t), # type: ignore
    hi: scalar_t, hj: scalar_t,
    dxi: vector(dtype=scalar_t, length=dim_t), dxj: vector(dtype=scalar_t, length=dim_t), # type: ignore
    dhi: scalar_t, dhj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
    useCRK: wp.bool,
    Ai: scalar_t, Bi: vector(length=dim_t, dtype=scalar_t), gradAi: vector(length=dim_t, dtype=scalar_t), gradBi: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
    dAi: scalar_t, dBi: vector(length=dim_t, dtype=scalar_t), dgradAi: vector(length=dim_t, dtype=scalar_t), dgradBi: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
):
    """JVP counterpart to `computeKernelGradientCRK`, dispatching between the
    CRK-corrected product rule (`correctGradientCRKJVP`) and the plain
    uncorrected kernel-gradient JVP (`kernels.kernelJVP.sphKernelGradientJVP`)
    on the same `useCRK` flag `computeKernelGradientCRK` itself branches on."""
    G, dG = sphKernelGradientJVP(xi, xj, hi, hj, dxi, dxj, dhi, dhj, kernelProperties, domainState)

    if useCRK:
        x_ij = computeDistanceVec(xi, xj, domainState)
        dx_ij = dxi - dxj
        W_ij, dW_ij = sphKernelJVP(xi, xj, hi, hj, dxi, dxj, dhi, dhj, kernelProperties, domainState)
        return correctGradientCRKJVP(
            W_ij, dW_ij, G, dG, x_ij, dx_ij,
            Ai, Bi, gradAi, gradBi, dAi, dBi, dgradAi, dgradBi,
            get_dim(xi),
        )
    return G, dG


@wp.func
def computeKernelGradientCRK(
    xi: vector(dtype=scalar_t, length=dim_t),
    xj: vector(dtype=scalar_t, length=dim_t),
    hi: scalar_t,
    hj: scalar_t,
    kernelProperties: kernelState,
    domainState: domainData,
    useCRK: wp.bool,
    Ai: scalar_t, Bi: vector(length=dim_t, dtype=scalar_t), gradAi: vector(length=dim_t, dtype=scalar_t), gradBi: matrix(shape=(dim_t, dim_t), dtype=scalar_t), # type: ignore
):
    x_ij = computeDistanceVec(xi, xj, domainState)
    kernelGradient = sphKernelGradient_ij(x_ij, hi, hj, kernelProperties, domainState)

    if useCRK:
        W_ij = sphKernel_ij(x_ij, hi, hj, kernelProperties, domainState)
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
    kernelProperties: kernelState,
    domainState: domainData,
    useCRK: wp.bool,
    Ai: scalar_t, Bi: vector(length=dim_t, dtype=scalar_t)
):
    x_ij = computeDistanceVec(xi, xj, domainState)
    w_ij = sphKernel_ij(x_ij, hi, hj, kernelProperties, domainState)
    if useCRK:
        xij = computeDistanceVec(xi, xj, domainState)
        return Ai * (scalar_t(1.0) + wp.dot(Bi, xij)) * w_ij
    return w_ij
