"""Order-p local polynomial fit (MLS/RKPM, LABFM): warp kernels.

Two neighbour-sum kernels share the canonical structured-kernel ABI prefix
(see crk/crk_moments.py, coreOperations/wp_interpolate.py):

* ``computePolyMoments_Kernel`` -- the packed lower triangle of the moment
  matrix ``M_i = sum_j w_ij P(xi_ij) P(xi_ij)^T`` for an arbitrary monomial
  basis (``exponents``, shape (n, dim)), ``w_ij = V_j W_ij``,
  ``xi_ij = (x_j - x_i) / h_i``.
* ``computePolyRHS_Kernel`` -- ``b_i = sum_j w_ij P(xi_ij) (f_j - c f_i)`` for a
  vector-valued field (``c = 1`` for the LABFM difference form, 0 for MLS).

The small dense solve ``c = M^-1 b`` and the derivative read-out live on the
torch side (``polyfit.py``): batched ``eigh``/pseudo-inverse, rank handling and
conditioning diagnostics. The pure-torch reference these are validated
against is ``higherOrderSPH/harness/rkpm.py``.

Monomials are evaluated in-kernel (``polyMonomial``) from the exponent table.
Both kernels use **one thread per output entry** (``N * L`` threads, L =
n(n+1)/2 for the moments, n * D for the right-hand side): each thread loops
the neighbours of its particle accumulating one scalar and stores it once into
a 2-D global output ``(N, L)``. Two alternatives were tried and rejected:

* a per-thread output vector of length L (~1000 for k = 8 in 2-D): the
  generated code took many minutes and several GB to compile;
* one thread per particle accumulating into the global output with ``+=``:
  Warp's reverse mode *replays* the forward body inside the adjoint kernel, so
  the read-modify-write re-adds to the saved forward output on every backward
  pass (gradients that grew linearly with the number of backward calls).

The cost is that each thread re-evaluates the pair weight (L-fold redundant
kernel evaluations); the weight is the cheap part of the sum at these sizes.
"""

import warp as wp
from typing import Any, Optional, Union
from warp.types import vector
import torch
from dataclasses import replace

from ..profiling import record_function
from ..type_config import *
from ..autograd import *
from ..dataTypes import *
from ..radiusSearch.grid_util import getIndexRangeLane, laneSum
from ..math import *
from ..kernels import *
from ..util import *
from ..enumTypes import *


@wp.func
def polyIntPow(x: scalar_t, n: wp.int32):
    """``x ** n`` for a non-negative integer ``n``, with an exactly-1 branch for
    ``n == 0`` (``wp.pow``'s adjoint would evaluate ``0 * x**-1``, NaN at x = 0)."""
    if n == 0:
        return scalar_t(1.0)
    return wp.pow(x, scalar_t(n))


@wp.func
def polyMonomial(e: Any, xi: Any, dim: wp.int32):
    """``prod_k xi_k ** e_k`` for the exponent vector ``e`` (dim <= 3).

    Deliberately NOT a loop over ``k`` or over the power: Warp's reverse mode
    gives wrong gradients for a variable that a dynamic loop updates
    multiplicatively (``r = r * x``) -- the d(position)/d(support) adjoints came
    out wrong by a non-constant factor while the mass/density ones were exact.
    Straight-line code over the (at most three) components has none of that.
    """
    r = polyIntPow(xi[0], e[0])
    if dim > 1:
        r = r * polyIntPow(xi[1], e[1])
    if dim > 2:
        r = r * polyIntPow(xi[2], e[2])
    return r


@wp.func
def polyFitWeight(
    iPtcl: Any, jPtcl: Any, domainState: domainData, kernelProperties: kernelState,
    correctionData: Any, j: wp.int32,
):
    """``(w_ij, xi_ij)`` for the pair: quadrature weight ``V_j W_ij`` and the
    scaled offset ``(x_j - x_i) / h_i``."""
    if correctionData.useVolume:
        vj = correctionData.referenceVolumes[j]
    else:
        vj = jPtcl.mass / jPtcl.density
    x_ij = computeDistanceVec(iPtcl.position, jPtcl.position, domainState)   # x_i - x_j
    w = vj * sphKernel_ij(x_ij, iPtcl.support, jPtcl.support, kernelProperties, domainState)
    xi = (-x_ij) / iPtcl.support
    return w, xi


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------

@wp.func
def computePolyMoments_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, referenceState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    iCorrectionData: Any, correctionData: Any,
    # end of the canonical ABI prefix

    ea: Any,  # type: ignore   exponents of the row basis function
    eb: Any,  # type: ignore   exponents of the column basis function
):
    s = scalar_t(0.0)
    count = wp.int32(0)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        w, xi = polyFitWeight(iPtcl, jPtcl, domainState, kernelProperties, correctionData, j)
        s += w * polyMonomial(ea, xi, dim) * polyMonomial(eb, xi, dim)
        # count only pairs inside the kernel support: the on-the-fly grid
        # traversal also visits the whole neighbouring cells, whose members
        # outside the support carry zero weight (and must not satisfy the
        # "enough neighbours for n basis functions" rank check)
        if w != scalar_t(0.0):
            count += 1
    return s, count


@wp.func
def computePolyMoments_Func_Adjacency(
    i: wp.int32, dim: wp.int32, lane: wp.int32, lanes: wp.int32,
    queryState: Any, referenceState: Any, correctionData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    # end of the canonical ABI prefix

    ea: Any,  # type: ignore
    eb: Any,  # type: ignore
):
    iPtcl = getParticleData(queryState, i)
    s = scalar_t(0.0)
    count = wp.int32(0)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return s, count
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)

    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, lane, lanes, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        ss, cc = computePolyMoments_Func_i(
            i, dim, iPtcl, referenceState, domainState, kernelProperties,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            iCorrectionData, correctionData,
            ea, eb,
        )
        s += ss
        count += cc
    return s, count


@wp.kernel
def computePolyMoments_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    exponents: wp.array(dtype=Any),  # type: ignore
    rowCol: wp.array(dtype=wp.vec2i),

    output_M: wp.array2d(dtype=scalar_t),  # type: ignore
    output_numNeighbors: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    L = rowCol.shape[0]
    i = tid // L
    l = tid - i * L
    if i >= queryState.positions.shape[0]:
        return
    rc = rowCol[l]
    s, count = computePolyMoments_Func_Adjacency(
        i, domainState.dim, 0, 1,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        exponents[rc[0]], exponents[rc[1]],
    )
    output_M[i, l] = s
    if l == 0:
        output_numNeighbors[i] = count


def _momentRowCol(exponents: torch.Tensor) -> torch.Tensor:
    n = exponents.shape[0]
    rows, cols = torch.tril_indices(n, n, device=exponents.device)
    return torch.stack([rows, cols], dim=1).to(torch.int32).contiguous()


_POLY_MOMENTS_SPEC = OperatorSpec(
    kernel=computePolyMoments_Kernel,
    outputs=(
        OutputSpec(dtype=scalar_t, shape=lambda ctx, extras: (
            ctx.query.positions.shape[0],
            (extras["exponents"].shape[0] * (extras["exponents"].shape[0] + 1)) // 2)),
        OutputSpec(dtype=wp.int32),
    ),
    extras=(ExtraSpec("exponents", ExtraKind.TENSOR),
            ExtraSpec("rowCol", ExtraKind.TENSOR)),
    numThreads=lambda ctx, extras: ctx.query.positions.shape[0] * extras["rowCol"].shape[0],
)


def _computePolyMoments_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    exponents: torch.Tensor,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency=None,
    referenceParticles: Optional[ParticleState] = None,
):
    """Packed lower-triangle moment matrices ``(N, n(n+1)/2)`` (row-major:
    entry ``a(a+1)/2 + b`` is ``M[a, b]``, ``b <= a``) and neighbour counts."""
    with record_function("warpSPH[PolyMoments]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(volumes=(queryVolumes, referenceVolumes)),
        )
        return launchOperator(_POLY_MOMENTS_SPEC, ctx, exponents=exponents,
                              rowCol=_momentRowCol(exponents))


# --------------------------------------------------------------------------
# right-hand side
# --------------------------------------------------------------------------

@wp.func
def computePolyRHS_Func_i(
    i: wp.int32, dim: wp.int32,
    iPtcl: Any, referenceState: Any,
    domainState: domainData, kernelProperties: kernelState,
    beginIndex: wp.int32, numIndices: wp.int32, offsetArray: wp.array(dtype=wp.int64),  # type: ignore
    iCorrectionData: Any, correctionData: Any,
    # end of the canonical ABI prefix

    ea: Any,  # type: ignore
    values: wp.array(dtype=Any),  # type: ignore
    centerValue: Any,  # type: ignore   f_i (zero for the MLS form)
    comp: wp.int32,
):
    s = scalar_t(0.0)
    for neighborIndex in range(numIndices):
        jj = beginIndex + neighborIndex
        j = wp.int32(offsetArray[jj])
        jPtcl = getParticleData(referenceState, j)
        if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
            if not checkDirectionality_j(jPtcl.kind, kernelProperties.operationMode):
                continue
        w, xi = polyFitWeight(iPtcl, jPtcl, domainState, kernelProperties, correctionData, j)
        fj = values[j]
        s += w * polyMonomial(ea, xi, dim) * (fj[comp] - centerValue[comp])
    return s


@wp.func
def computePolyRHS_Func_Adjacency(
    i: wp.int32, dim: wp.int32, lane: wp.int32, lanes: wp.int32,
    queryState: Any, referenceState: Any, correctionData: Any,
    domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData, numOffsets: wp.int32,
    kernelProperties: kernelState,
    # end of the canonical ABI prefix

    ea: Any,  # type: ignore
    values: wp.array(dtype=Any),  # type: ignore
    subtractCenter: wp.int32,
    comp: wp.int32,
):
    iPtcl = getParticleData(queryState, i)
    s = scalar_t(0.0)
    if kernelProperties.operationMode != wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(iPtcl.kind, kernelProperties.operationMode):
            return s
    iCorrectionData = getParticleCorrectionData_i(correctionData, i)
    centerValue = values[i] * scalar_t(0.0)
    if subtractCenter != 0:
        centerValue = values[i]

    for o in range(numOffsets):
        beginIndex, numIndices = getIndexRangeLane(i, o, lane, lanes, useAdjacency, adjacencyState, gridState, queryState, domainState)
        if beginIndex < 0:
            continue
        s += computePolyRHS_Func_i(
            i, dim, iPtcl, referenceState, domainState, kernelProperties,
            beginIndex, numIndices, adjacencyState.neighborList if useAdjacency else gridState.sortIndex,
            iCorrectionData, correctionData,
            ea, values, centerValue, comp,
        )
    return s


@wp.kernel
def computePolyRHS_Kernel(
    queryState: Any, referenceState: Any, domainState: domainData,
    useAdjacency: wp.bool, adjacencyState: adjacencyData, gridState: gridData,
    correctionData: Any, kernelProperties: kernelState,
    # canonical ABI prefix -- do not change

    exponents: wp.array(dtype=Any),  # type: ignore
    values: wp.array(dtype=Any),  # type: ignore
    subtractCenter: wp.int32,
    numComponents: wp.int32,

    output_B: wp.array2d(dtype=scalar_t),  # type: ignore
):
    tid = wp.tid()
    width = exponents.shape[0] * numComponents
    i = tid // width
    l = tid - i * width
    if i >= queryState.positions.shape[0]:
        return
    a = l // numComponents
    comp = l - a * numComponents
    output_B[i, l] = computePolyRHS_Func_Adjacency(
        i, domainState.dim, 0, 1,
        queryState, referenceState, correctionData, domainState,
        useAdjacency, adjacencyState, gridState, gridState.numOffsets if not useAdjacency else 1,
        kernelProperties,
        exponents[a], values, subtractCenter, comp,
    )


_POLY_RHS_SPEC = OperatorSpec(
    kernel=computePolyRHS_Kernel,
    outputs=(OutputSpec(dtype=scalar_t, shape=lambda ctx, extras: (
        ctx.query.positions.shape[0],
        extras["exponents"].shape[0] * int(extras["numComponents"]))),),
    extras=(ExtraSpec("exponents", ExtraKind.TENSOR),
            ExtraSpec("values", ExtraKind.TENSOR),
            ExtraSpec("subtractCenter", ExtraKind.SCALAR),
            ExtraSpec("numComponents", ExtraKind.SCALAR)),
    numThreads=lambda ctx, extras: (ctx.query.positions.shape[0]
                                    * extras["exponents"].shape[0] * int(extras["numComponents"])),
)


def _computePolyRHS_stateBackend(
    queryParticles: ParticleState,
    operationProperties: OperationProperties,
    domain: DomainDescription,
    exponents: torch.Tensor,
    values: torch.Tensor,
    subtractCenter: bool,
    queryVolumes: Optional[torch.Tensor] = None, referenceVolumes: Optional[torch.Tensor] = None,
    adjacency=None,
    referenceParticles: Optional[ParticleState] = None,
):
    """``b_i`` of shape ``(N, n * D)`` (entry ``a * D + c``) for a field of
    shape ``(N, D)``. ``values`` must already be 2-D."""
    with record_function("warpSPH[PolyRHS]"):
        ctx = SPHContext(
            query=queryParticles, properties=operationProperties, domain=domain,
            adjacency=adjacency, reference=referenceParticles,
            corrections=Corrections(volumes=(queryVolumes, referenceVolumes)),
        )
        return launchOperator(
            _POLY_RHS_SPEC, ctx, exponents=exponents, values=values,
            subtractCenter=int(subtractCenter), numComponents=int(values.shape[1]),
        )
