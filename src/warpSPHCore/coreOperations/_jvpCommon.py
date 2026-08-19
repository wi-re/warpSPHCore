"""Shared boilerplate for the pair-indexed Tier-2 JVP kernels
(`warpier_tier2_operators_plan.md` Steps 0/3): SoA/domain/kernel-state
builders extracted from `wp_densityJVP.py` unchanged, plus the shared
per-pair `(W_ij, dW_ij)` kernel launcher every `wp_<op>JVP.py` needs (Step 3
factors this out of `wp_densityJVP.py` so `wp_interpolateJVP.py` and later
operators reuse it instead of re-launching their own copy).
"""

from typing import Any, Optional
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarp, castTorchToWarpAsBuiltins, allocateTorchWarp
from ..enumTypes import supportSchemeToUint
from ..kernels.kernelJVP import sphKernelJVP
from ..util.stateUtil import getParticle

__all__ = ['buildParticleSoA', 'buildDomainState', 'buildKernelState', 'launchPairKernelJVP', 'gradientWeights']

_SoA_BY_DIM = {1: particleDataSoA_1, 2: particleDataSoA_2, 3: particleDataSoA_3}


def buildParticleSoA(
    dim: int,
    positions: torch.Tensor,
    supports: torch.Tensor,
    masses: torch.Tensor,
    densities: Optional[torch.Tensor] = None,
):
    SoA = _SoA_BY_DIM[dim]()
    SoA.positions = castTorchToWarpAsBuiltins(positions.contiguous())
    SoA.supports = castTorchToWarp(supports.contiguous())
    SoA.masses = castTorchToWarp(masses.contiguous())
    n = positions.shape[0]
    if densities is None:
        densities = torch.zeros(n, device=positions.device, dtype=positions.dtype)
    SoA.densities = castTorchToWarp(densities.contiguous())
    SoA.kinds = castTorchToWarp(torch.zeros(n, device=positions.device, dtype=torch.int32))
    return SoA


def buildDomainState(domain: DomainDescription) -> domainData:
    d = domainData()
    d.domainMin = castTorchToWarp(domain.min)
    d.domainMax = castTorchToWarp(domain.max)
    d.periodicity = castTorchToWarp(domain.periodic)
    d.dim = domain.dim
    return d


def buildKernelState(kernel: KernelFunctions, supportMode: SupportScheme) -> kernelState:
    k = kernelState()
    k.kernelFunction = kernel.value
    k.supportMode = supportSchemeToUint(supportMode)
    return k


@wp.kernel
def _sphKernelJVP_PairKernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    edgeI: wp.array(dtype=wp.int64),
    edgeJ: wp.array(dtype=wp.int64),
    outW: wp.array(dtype=scalar_t),
    outDW: wp.array(dtype=scalar_t),
):
    e = wp.tid()
    if e >= edgeI.shape[0]:
        return
    i = wp.int32(edgeI[e])
    j = wp.int32(edgeJ[e])

    xi, hi, _mi, _rhoi, _ki = getParticle(queryState, i)
    xj, hj, _mj, _rhoj, _kj = getParticle(referenceState, j)
    dxi, dhi, _dmi, _drhoi, _dki = getParticle(queryTangentState, i)
    dxj, dhj, _dmj, _drhoj, _dkj = getParticle(referenceTangentState, j)

    W, dW = sphKernelJVP(xi, xj, hi, hj, dxi, dxj, dhi, dhj, kernelProperties, domainState)
    outW[e] = W
    outDW[e] = dW


def launchPairKernelJVP(
    queryState, referenceState, queryTangentState, referenceTangentState,
    domainState: domainData, kernelProperties: kernelState,
    edgeI, edgeJ,
):
    """One thread per adjacency pair `(i, j)`, producing the pairwise
    `(W_ij, dW_ij)` JVP as flat `[numPairs]` torch tensors -- the single
    warp kernel launch shared by every Tier-2 `wp_<op>JVP.py` operator
    (`warpier_tier2_operators_plan.md` Step 3). `edgeI`/`edgeJ` are the
    already-cast warp int64 arrays (`castTorchToWarp(adjacency.i/.j)`).
    """
    numPairs = edgeI.shape[0]
    W_t, W_w = allocateTorchWarp(numPairs, scalar_t, edgeI.device)
    dW_t, dW_w = allocateTorchWarp(numPairs, scalar_t, edgeI.device)

    wp.launch(
        _sphKernelJVP_PairKernel,
        dim=numPairs,
        inputs=[queryState, referenceState, queryTangentState, referenceTangentState,
                domainState, kernelProperties, edgeI, edgeJ, W_w, dW_w],
        device=edgeI.device,
    )
    return W_t, dW_t


def gradientWeights(
    massJ: torch.Tensor, densityI: torch.Tensor, densityJ: torch.Tensor,
    dMassJ: torch.Tensor, dDensityI: torch.Tensor, dDensityJ: torch.Tensor,
    scheme: GradientScheme,
):
    """`coeff_ij = fi*A_ij + fj*B_ij` (`fi`/`fj` frozen -- Tier 1 territory,
    contribute no term of their own here), `A`/`B`/`dA`/`dB` per
    `warpier_adjoint.md` Tier 2.2. Shared by Gradient/Divergence/Curl (their
    own `coeff_ij`) and Laplacian(Brookshaw) (`B` doubles as `q_ij`'s
    coefficient, `warpier_adjoint.md` Tier 2.2 finding 2 -- not a coincidence,
    not re-derived independently). All six inputs and both outputs are flat
    `[numPairs]` (already indexed by `adjacency.i`/`.j`), unlike the
    `scripts/spike_forward_mode_tier2_gradient.py` this ports, which worked
    on the dense `(n,n)` all-pairs grid -- the formulas are otherwise
    identical."""
    Vj = massJ / densityJ
    dVj = dMassJ / densityJ - massJ * dDensityJ / densityJ ** 2

    if scheme == GradientScheme.Naive:
        A, dA = torch.zeros_like(Vj), torch.zeros_like(Vj)
        B, dB = Vj, dVj
    elif scheme == GradientScheme.Difference:
        A, dA = -Vj, -dVj
        B, dB = Vj, dVj
    elif scheme == GradientScheme.Summation:
        A, dA = Vj, dVj
        B, dB = Vj, dVj
    elif scheme == GradientScheme.Symmetric:
        A = massJ / densityI
        dA = dMassJ / densityI - massJ * dDensityI / densityI ** 2
        B = massJ * densityI / densityJ ** 2
        dB = (dMassJ * densityI / densityJ ** 2
              + massJ * dDensityI / densityJ ** 2
              - 2.0 * massJ * densityI * dDensityJ / densityJ ** 3)
    else:
        raise ValueError(f"gradientWeights: unsupported GradientScheme {scheme}")
    return A, B, dA, dB
