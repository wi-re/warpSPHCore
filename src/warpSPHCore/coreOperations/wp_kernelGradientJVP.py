"""Shared per-pair `(G_ij, dG_ij)` JVP kernel launcher for Gradient/
Divergence/Curl/Laplacian(Brookshaw) (`warpier_tier2_operators_plan.md`
Step 3, `warpier_adjoint.md` Tier 2.2): all four operators route through the
same `sphKernelGradient_ij` building block with `useCRK=False`
(CRK/renormalization are out of Tier-2 scope) -- only the downstream
coefficient/combination step differs per operator, which is pure torch, not
warp. So there is exactly ONE new warp kernel launch here, shared by every
`wp_<op>JVP.py` file that needs `nabla_i W_ij` and its JVP.
"""

from typing import Any
import torch
import warp as wp

from ..type_config import *
from ..dataTypes import *
from ..enumTypes import *
from ..util import castTorchToWarpAsBuiltins
from ..kernels.kernelJVP import sphKernelGradientJVP
from ..util.stateUtil import getParticle

__all__ = ['launchPairKernelGradientJVP']


@wp.kernel
def _sphKernelGradientJVP_PairKernel(
    queryState: Any,
    referenceState: Any,
    queryTangentState: Any,
    referenceTangentState: Any,
    domainState: domainData,
    kernelProperties: kernelState,
    edgeI: wp.array(dtype=wp.int64),
    edgeJ: wp.array(dtype=wp.int64),
    outG: wp.array(dtype=Any),  # type: ignore
    outDG: wp.array(dtype=Any),  # type: ignore
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

    G, dG = sphKernelGradientJVP(xi, xj, hi, hj, dxi, dxj, dhi, dhj, kernelProperties, domainState)
    outG[e] = G
    outDG[e] = dG


def launchPairKernelGradientJVP(
    queryState, referenceState, queryTangentState, referenceTangentState,
    domainState: domainData, kernelProperties: kernelState,
    edgeI, edgeJ, dim: int, device, dtype,
):
    """One thread per adjacency pair, producing the pairwise `(G_ij, dG_ij)`
    JVP (`d(nabla_i W_ij)/d{x,h}`) as flat `[numPairs, dim]` torch tensors --
    the single warp kernel launch Gradient/Divergence/Curl/
    Laplacian(Brookshaw) all share. `edgeI`/`edgeJ` are the already-cast warp
    int64 arrays (`castTorchToWarp(adjacency.i/.j)`), same convention as
    `_jvpCommon.launchPairKernelJVP`. `dim`/`device`/`dtype` are the torch
    (not warp) vector width/device/dtype -- the output kernel arguments are
    declared generic (`wp.array(dtype=Any)`, matching every other
    dimension-generic production kernel, e.g. `wp_gradient.py`'s
    `outputValues`) rather than using `type_config.vec_t` directly, since
    `vec_t` resolves to a length-`Any` (ungrounded) vector type whenever
    `warpSPHCore_DIM` isn't pinned via env var -- concrete per-call shape
    comes from `castTorchToWarpAsBuiltins` on an actual `(numPairs, dim)`
    torch tensor instead, the same way `buildParticleSoA` resolves
    `positions`' vector width from the tensor it's given rather than a
    fixed constant.
    """
    numPairs = edgeI.shape[0]
    G_t = torch.zeros((numPairs, dim), device=device, dtype=dtype)
    dG_t = torch.zeros((numPairs, dim), device=device, dtype=dtype)
    G_w = castTorchToWarpAsBuiltins(G_t)
    dG_w = castTorchToWarpAsBuiltins(dG_t)

    wp.launch(
        _sphKernelGradientJVP_PairKernel,
        dim=numPairs,
        inputs=[queryState, referenceState, queryTangentState, referenceTangentState,
                domainState, kernelProperties, edgeI, edgeJ, G_w, dG_w],
        device=edgeI.device,
    )
    return G_t, dG_t
