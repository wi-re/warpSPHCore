"""``OperatorSpec`` / ``SPHContext`` / ``launchOperator``: the declared
operator ABI (warpier_fields.md Section 8, Step I).

Formalizes what every ``warpWrapper2`` call site already does by
convention -- a fixed 10-slot positional state tuple, None-padded per call;
an ``additionalArguments`` tuple re-analysed by isinstance every launch; an
output dtype/shape probed inline at every call site -- into a spec declared
once per kernel (``OperatorSpec``) and a context built once per call
(``SPHContext``).

Not a new caching layer and not a behaviour change: ``launchOperator``
resolves straight into the same ``_launch`` engine ``warpWrapper2`` already
uses (``wrapper.py``), so a kernel ported to this ABI and the
``warpWrapper2`` shim it replaces launch identically. See Section 8.4's
migration plan.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from ..dataTypes import (
    AdjacencyListWarp,
    CompactHashMap,
    CRKState,
    CRKTangentState,
    DomainDescription,
    ExecutionMode,
    GradHState,
    OperationProperties,
    ParticleState,
    ParticleTangentState,
    RenormalizationState,
    RenormalizationTangentState,
)
from .launcher import launch_kernel
from .stateAwareWarpFunction import hasLiveTangent
from .wrapper import _launch


class ShapeOf(Enum):
    """Output-shape sources ``OutputSpec.shape`` accepts besides a literal
    int or a resolver callable. ``QUERY`` is the only one any operator needs
    today -- every kernel in this codebase launches one thread per query
    particle."""

    QUERY = 0


@dataclass(frozen=True)
class OutputSpec:
    """One output slot's dtype and shape. Both accept either a literal
    value or a ``Callable[[SPHContext, dict], Any]`` resolver.

    Section 8.3 frames output dtypes as "resolved at import" -- true for
    operators whose output dtype depends only on dim/precision (Density,
    Covariance, the CRK family), **not** true for the operators
    (Interpolate, Gradient, Divergence, Curl, Laplacian) whose output dtype
    packs the *runtime* shape of a queryValues/referenceValues tensor that
    isn't known until call time. For those, ``dtype`` is a resolver,
    declared once here instead of hand-inlined at each call site -- the
    call-time cast survives (nothing here turns an unknown-until-called
    shape into a compile-time constant), but the duplication across call
    sites does not. Recorded here rather than silently working around it,
    per this plan's habit of correcting its own sketch against what the
    code actually does.
    """

    dtype: Union[Any, Callable[["SPHContext", Dict[str, Any]], Any]]
    shape: Union[int, ShapeOf, Callable[["SPHContext", Dict[str, Any]], Any]] = ShapeOf.QUERY

    def resolve(self, ctx: "SPHContext", extras: Dict[str, Any]) -> Tuple[Any, Any]:
        # Plain `callable(...)` misfires here: a literal Warp dtype (e.g.
        # `scalar_t`, `wp.int32`) is itself a class, and classes are
        # callable -- `callable(scalar_t)` is True, so a naive check would
        # try to invoke the dtype as a resolver and fail (or, worse, invoke
        # it and silently accept whatever it returns). A resolver is always
        # a plain function/lambda declared alongside the OperatorSpec, never
        # a type, so `types.FunctionType` is the correct discriminator.
        shape = (
            self.shape(ctx, extras)
            if isinstance(self.shape, types.FunctionType)
            else (ctx.query.positions.shape[0] if self.shape is ShapeOf.QUERY else self.shape)
        )
        dtype = (
            self.dtype(ctx, extras)
            if isinstance(self.dtype, types.FunctionType)
            else self.dtype
        )
        return shape, dtype


class ExtraKind(Enum):
    TENSOR = 0
    SCALAR = 1


@dataclass(frozen=True)
class ExtraSpec:
    """One declared name in an operator's ``additionalArguments``. ``kind``
    is validated against what's actually passed at each call (Section 8.1's
    "untyped, re-analysed every call" complaint) -- the ABI is a contract,
    checked at the boundary, not a convention every call site has to get
    right on its own."""

    name: str
    kind: ExtraKind


class ThreadSpec(Enum):
    """wp.launch thread-count sources. ``QUERY_COUNT`` (the common case)
    leaves ``numThreads`` unset, so ``_launch`` defaults to the first
    output's resolved shape -- exactly today's behaviour. Some kernels
    (e.g. compSPH's pair-indexed balance term) launch one thread per query
    particle while writing an output shaped by the *pair* count, so the
    first-output-shape fallback would resolve to the wrong thread count for
    them -- ``OperatorSpec.numThreads`` is the escape hatch for that case
    (Section 8.4, Step J)."""

    QUERY_COUNT = 0


@dataclass(frozen=True)
class JVPSpec:
    """Opts a kernel into ``StateAwareWarpFunction.jvp()`` dispatch
    (``warpier_unified_operator_wrapper_plan.md``): a dual-tensor argument
    reaching this kernel's launch is turned into a tangent output by
    delegating to ``warpOperationJVP`` -- the same, already-exhaustively-
    tested dispatch/gating logic (CRK/renormalization scope tables,
    Laplacian-scheme restrictions, Divergence/Curl restrictions, Density's
    and Covariance's own narrower scopes) a caller gets from an explicit
    ``warpOperationJVP`` call, reused rather than re-derived here.

    ``queryValueExtra``/``referenceValueExtra`` name the declared
    ``OperatorSpec.extras`` entries (if any) eligible for a Tier-1 value
    tangent -- Interpolate: ``referenceValueExtra="referenceValues"`` only
    (its kernel never reads ``queryValues``); Gradient/Divergence/Curl/
    Laplacian: ``queryValueExtra="queryValuesFlat"``,
    ``referenceValueExtra="referenceValuesFlat"``; Density/Covariance: both
    ``None`` (geometry-tangent only, no value input at all). A live tangent
    anywhere else in the call this ``JVPSpec`` cannot account for (domain
    bounds, grid-traversal structure) still raises ``NotImplementedError``
    rather than silently dropping it.
    """

    queryValueExtra: Optional[str] = None
    referenceValueExtra: Optional[str] = None


@dataclass(frozen=True)
class OperatorSpec:
    """Declared once per kernel, at import time, next to the kernel itself.
    See warpier_fields.md Section 8.2.

    ``numThreads``, when set, is a resolver called as ``(ctx, extras) ->
    int`` and passed through to ``_launch`` explicitly -- for the rare
    kernel whose thread count is not the shape of its first output (see
    ``ThreadSpec``'s docstring). ``None`` (the default) preserves today's
    behaviour: ``_launch`` derives it from the first output's shape.

    ``jvp``, when set, opts this kernel into ``StateAwareWarpFunction``'s
    ``jvp()`` dispatch (``warpier_unified_operator_wrapper_plan.md``).
    ``None`` (the default) preserves today's behaviour exactly -- a
    dual-tensor argument reaching this kernel's launch raises a clear error
    rather than silently returning a tangent-free dual output.
    """

    kernel: Any
    outputs: Tuple[OutputSpec, ...]
    extras: Tuple[ExtraSpec, ...] = ()
    threads: ThreadSpec = ThreadSpec.QUERY_COUNT
    numThreads: Optional[Callable[["SPHContext", Dict[str, Any]], int]] = None
    jvp: Optional[JVPSpec] = None


@dataclass
class Corrections:
    """Collapses 5 of the 10 ``defaultStateArguments`` slots (Section
    8.2)."""

    volumes: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
    crk: Union[CRKState, Tuple[CRKState, CRKState], None] = None
    gradH: Optional[GradHState] = None
    renorm: Optional[RenormalizationState] = None


EMPTY_CORRECTIONS = Corrections()


@dataclass
class SPHContext:
    """Per-call, caller-held handle -- named, non-positional state (Section
    8.2/8.3), replacing the None-padded 10-tuple.

    ``mode`` defaults to ``AUTO``; ``launchOperator`` rejects ``FORWARD``
    explicitly (declared and deliberately unimplemented, same as
    ``structFor``/``getStateBundle``) and otherwise passes ``mode`` through
    unresolved -- nothing downstream branches on ``NONE`` vs ``REVERSE`` yet
    (``structFor``'s table maps both to the same struct rows), so resolving
    ``AUTO`` to a concrete value has no behaviour to attach to until
    something does.
    """

    query: ParticleState
    properties: OperationProperties
    domain: DomainDescription
    adjacency: Optional[Union[AdjacencyListWarp, CompactHashMap]] = None
    reference: Optional[ParticleState] = None
    corrections: Corrections = field(default_factory=Corrections)
    mode: ExecutionMode = ExecutionMode.AUTO


# extractStateInfo's fixed 36-slot flat-tensor layout (arg_extract.py's own
# docstring) -- stable, documented positions this module reads directly
# rather than re-deriving. Positions not named here (8/9 kinds; 11/12
# grad-h omegas; 19-22 the reference-role CRK slot, which -- since every
# current operator's stateBackend accepts only a single CRKState shared by
# both roles, never a query/reference pair -- always aliases the same tensor
# objects as 15-18, so torch's per-object tangent lookup reports identical
# values at both and only 15-18 need reading; 23-32/35 adjacency/grid
# traversal) are never real Tensor positions with a meaningful tangent, or
# are read via a primal object captured by closure instead (see
# ``_build_geometry_jvp_fn``).
_QPOS, _RPOS, _QSUP, _RSUP, _QMAS, _RMAS, _QDEN, _RDEN = range(8)
_RENORM_MAT = 10
_QVOL, _RVOL = 13, 14
_QCRK_A, _QCRK_B, _QCRK_GRADA, _QCRK_GRADB = 15, 16, 17, 18
_STATE_N = 36
# grid_qMin/qMax, domainMin/domainMax: the one case with no warpOperationJVP
# kwarg to naturally reject a live tangent here (every other structural
# position either can't carry a float tangent at all -- kinds, adjacency
# index arrays are integer -- or maps to a primal object warpOperationJVP
# itself already validates, e.g. gradHState/queryVolumes). No operator's
# geometry JVP differentiates w.r.t. domain bounds or grid structure, so
# this is an explicit guard rather than a silent drop.
_NO_TANGENT_HOME = (27, 28, 33, 34)


def _unflattenValue(spec: OperatorSpec, extras: Dict[str, Any], flatTensor: Optional[torch.Tensor]):
    """Gradient/Divergence/Curl/Laplacian's ``queryValuesFlat``/
    ``referenceValuesFlat`` extras are `.view(-1, flatInputShape)` of the
    caller's original ``queryValues``/``referenceValues`` (``_computeSPH*
    _stateBackend``'s own preprocessing) -- a no-op reshape for a scalar or
    already-rank-1 (vector) field, but for a genuinely scalar field
    (``inputShape == ()``) it turns ``[N]`` into ``[N, 1]``, which
    ``warpOperationJVP``/``computeSPH<Op>GeometryJVP`` (same unflattened
    shape contract as the public ``warpOperation`` API) do not expect --
    inverted here via the ``numDims`` scalar extra every one of those four
    operators also declares. Interpolate declares no ``numDims`` extra at
    all (it reshapes only for rank>2 fields, via a different mechanism) --
    passed through unchanged for it, and for any operator with no such
    extra.
    """
    if flatTensor is None:
        return None
    extraNames = {e.name for e in spec.extras}
    if "numDims" not in extraNames:
        return flatTensor
    numDims = int(extras["numDims"])
    if numDims == 0:
        return flatTensor.view(-1)
    if numDims == 1:
        return flatTensor
    raise NotImplementedError(
        "StateAwareWarpFunction.jvp: value-tangent reconstruction for rank>=2 fields "
        "(matrix or higher) is not supported by this bridge -- the flattened shape "
        "cannot be inverted without the field's original per-dimension shape, which "
        "OperatorSpec.extras does not carry. Call warpOperationJVP directly for this case."
    )


def _reflattenResult(spec: OperatorSpec, extras: Dict[str, Any], result: torch.Tensor) -> torch.Tensor:
    """``StateAwareWarpFunction.forward()`` returns the kernel's *raw*
    output -- shape ``[N, flatOutputShape]``, matching the warp dtype
    ``_get_warp_vector_dtype(flatOutputShape, ...)`` resolved for it -- and
    only each operator's own ``_computeSPH<Op>_stateBackend`` wrapper
    reshapes that to the public ``[N, *outputShape]`` shape, *after*
    ``launchOperator`` (and so ``StateAwareWarpFunction``) has already
    returned. ``jvp_fn`` delegates to ``warpOperationJVP``, which goes
    through that same public (already-reshaped) path -- so its result must
    be reshaped back to the raw flat form before ``jvp()`` hands it to
    torch, which checks the tangent against ``forward()``'s raw output
    shape specifically. A no-op for Interpolate/Density/Covariance, whose
    raw and public shapes already coincide (no ``flatOutputShape`` extra).
    Found the hard way: ``outputShape`` only equals ``(flatOutputShape,)``
    for Gradient (which always appends exactly one spatial dimension) --
    Laplacian (``outputShape = inputShape``, rank 0 for a scalar field) and
    Divergence (``outputShape = inputShape[:-1]``, also rank 0 for a vector
    field) both collapse a dimension the flat form still carries, and hit
    this mismatch even though Gradient never does.
    """
    extraNames = {e.name for e in spec.extras}
    if "flatOutputShape" not in extraNames:
        return result
    flatOutputShape = int(extras["flatOutputShape"])
    return result.reshape(-1, flatOutputShape)


def _extraFlatPosition(spec: OperatorSpec, extras: Dict[str, Any], name: str) -> int:
    """Maps a declared ``OperatorSpec.extras`` name to its position in
    ``_launch``'s flat tensor list: ``additionalArguments`` (built in
    ``spec.extras`` order) keeps only the ``ExtraKind.TENSOR`` entries when
    flattened (``_launch``'s own ``add_tensor_pos``), appended after the
    36-slot state prefix -- mirrored here rather than threaded out of
    ``_launch``, since ``launchOperator`` already has ``spec``/``extras`` in
    scope to compute it directly."""
    tensorExtraNames = [e.name for e in spec.extras if e.kind is ExtraKind.TENSOR]
    return _STATE_N + tensorExtraNames.index(name)


def _build_geometry_jvp_fn(spec: OperatorSpec, sphCtx: SPHContext, extras: Dict[str, Any]):
    """warpier_unified_operator_wrapper_plan.md Phase 2. Builds the
    ``jvp_fn`` closure ``StateAwareWarpFunction.jvp()`` delegates to for any
    ``OperatorSpec.jvp``-declaring kernel, covering the Tier-1 value tangent,
    the Tier-2 geometry tangent, and their sum (when both are live) in one
    path by delegating entirely to ``warpOperationJVP`` -- not re-deriving
    any of its dispatch/gating logic here.

    Captures *sphCtx* (this call's ``SPHContext``) and *extras* by closure,
    so every PRIMAL semantic object (``ParticleState``, ``CRKState``,
    ``RenormalizationState``, ``adjacency``, ``domain``, the primal
    ``queryValues``/``referenceValues`` tensors) is the caller's own
    original Python object -- never reconstructed from flat tensors. Only
    the TANGENT objects need reconstructing, from ``flat_tangents``'s
    positions, using ``extractStateInfo``'s fixed layout (the ``_Q*``/``_R*``
    constants above).
    """
    queryValuePos = (
        _extraFlatPosition(spec, extras, spec.jvp.queryValueExtra)
        if spec.jvp.queryValueExtra else None
    )
    referenceValuePos = (
        _extraFlatPosition(spec, extras, spec.jvp.referenceValueExtra)
        if spec.jvp.referenceValueExtra else None
    )

    def jvp_fn(autogradCtx, flat_tangents):
        # Reuse the liveness mask StateAwareWarpFunction.jvp() computed and
        # stashed on the autograd ctx -- every hasLiveTangent is a GPU->CPU
        # sync and the positions below are consulted repeatedly (field() plus
        # the _NO_TANGENT_HOME guard), so re-checking would re-pay that sync
        # per call site. Fall back to computing it if absent (jvp_fn is only
        # ever invoked through jvp(), which always sets it; this keeps a
        # direct call well-defined).
        live_mask = getattr(autogradCtx, "_jvp_live_mask", None)
        n = len(flat_tangents)
        if live_mask is None:
            live_mask = [hasLiveTangent(t) for t in flat_tangents]

        def isLive(i):
            if i is None or i < 0 or i >= n:
                return False
            return live_mask[i]

        for i in _NO_TANGENT_HOME:
            if isLive(i):
                raise NotImplementedError(
                    "StateAwareWarpFunction.jvp: a live (non-zero) tangent on a domain-bounds "
                    f"or grid-traversal argument (flat position {i}) has no JVP formula -- no "
                    "operator's geometry JVP differentiates w.r.t. domain bounds or grid "
                    "structure."
                )

        def field(i):
            return flat_tangents[i] if isLive(i) else None

        queryTangentState = ParticleTangentState(
            positions=field(_QPOS), supports=field(_QSUP), masses=field(_QMAS), densities=field(_QDEN),
        )
        referenceTangentState = ParticleTangentState(
            positions=field(_RPOS), supports=field(_RSUP), masses=field(_RMAS), densities=field(_RDEN),
        )

        crkTangentState = None
        if sphCtx.corrections.crk is not None:
            crkFields = (field(_QCRK_A), field(_QCRK_B), field(_QCRK_GRADA), field(_QCRK_GRADB))
            if any(f is not None for f in crkFields):
                # CRKTangentState has no Optional fields (a caller supplying CRK tangent
                # support supplies all four) -- substitute a primal-shaped zero for
                # whichever of the four has no live tangent of its own.
                crk = sphCtx.corrections.crk
                crkTangentState = CRKTangentState(
                    A=crkFields[0] if crkFields[0] is not None else torch.zeros_like(crk.A),
                    B=crkFields[1] if crkFields[1] is not None else torch.zeros_like(crk.B),
                    gradA=crkFields[2] if crkFields[2] is not None else torch.zeros_like(crk.gradA),
                    gradB=crkFields[3] if crkFields[3] is not None else torch.zeros_like(crk.gradB),
                )

        renormalizationTangentState = None
        renormMatTangent = field(_RENORM_MAT)
        if sphCtx.corrections.renorm is not None and renormMatTangent is not None:
            renormalizationTangentState = RenormalizationTangentState(renormalizationMatrices=renormMatTangent)

        tangentQueryValues = _unflattenValue(
            spec, extras, field(queryValuePos) if queryValuePos is not None else None,
        )
        tangentReferenceValues = _unflattenValue(
            spec, extras, field(referenceValuePos) if referenceValuePos is not None else None,
        )
        primalQueryValues = _unflattenValue(
            spec, extras, extras[spec.jvp.queryValueExtra] if spec.jvp.queryValueExtra else None,
        )
        primalReferenceValues = _unflattenValue(
            spec, extras, extras[spec.jvp.referenceValueExtra] if spec.jvp.referenceValueExtra else None,
        )
        # An operator needing BOTH queryValues/referenceValues (Gradient/
        # Divergence/Curl/Laplacian -- unlike Interpolate, which only ever
        # reads referenceValues) rejects a bare None on either side
        # (warpOperation's own validation). Only one side carrying a live
        # tangent still means "a value tangent is being requested" -- the
        # other side's contribution is exactly zero, matching Phase 1's own
        # zeros_like fallback for the equivalent case.
        if tangentQueryValues is not None or tangentReferenceValues is not None:
            if spec.jvp.queryValueExtra and tangentQueryValues is None:
                tangentQueryValues = torch.zeros_like(primalQueryValues)
            if spec.jvp.referenceValueExtra and tangentReferenceValues is None:
                tangentReferenceValues = torch.zeros_like(primalReferenceValues)

        queryVolumes, referenceVolumes = sphCtx.corrections.volumes

        # local import: operations.py imports coreOperations, which imports
        # autograd -- importing warpOperationJVP at module scope here would
        # be circular. By the time jvp_fn is actually invoked (runtime, not
        # import time), every module involved is already fully loaded.
        from ..operations import warpOperationJVP

        result = warpOperationJVP(
            sphCtx.query, sphCtx.properties, sphCtx.domain,
            tangentQueryValues=tangentQueryValues, tangentReferenceValues=tangentReferenceValues,
            queryTangentState=queryTangentState, referenceTangentState=referenceTangentState,
            queryValues=primalQueryValues, referenceValues=primalReferenceValues,
            queryVolumes=queryVolumes, referenceVolumes=referenceVolumes,
            tangentReferenceVolumes=field(_RVOL),
            adjacency=sphCtx.adjacency,
            referenceParticles=sphCtx.reference,
            crkState=sphCtx.corrections.crk, crkTangentState=crkTangentState,
            gradHState=sphCtx.corrections.gradH,
            renormalizationState=sphCtx.corrections.renorm,
            renormalizationTangentState=renormalizationTangentState,
        )
        result = _reflattenResult(spec, extras, result)
        if len(spec.outputs) > 1:
            # Multi-output kernels (Covariance: matrix + numNeighbors) --
            # warpOperationJVP only ever returns the differentiable output's
            # tangent; every other output (an integer neighbor count here)
            # has none, mirroring how backward() never seeds a grad for it
            # either. jvp() must return one entry per forward() output.
            return (result,) + (None,) * (len(spec.outputs) - 1)
        return result

    return jvp_fn


def launchOperator(
    spec: OperatorSpec, ctx: SPHContext, **extras
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """The one entry point for the declared operator ABI (Section 8.2).

    Resolves *spec* against *ctx* into exactly the call shape
    ``warpWrapper2`` builds by hand, then delegates to the same ``_launch``
    engine -- no behaviour differs from an equivalent ``warpWrapper2`` call,
    only how a caller arrives at the arguments.
    """
    if ctx.mode is ExecutionMode.FORWARD:
        raise NotImplementedError(
            "ExecutionMode.FORWARD is not implemented (warpier_fields.md "
            "Section 3.6 / Step G) -- rejected at the SPHContext boundary, "
            "same as structFor/getStateBundle reject it internally."
        )

    declared = {e.name: e for e in spec.extras}
    missing = [name for name in declared if name not in extras]
    if missing:
        raise TypeError(f"launchOperator: missing declared extra(s): {missing}")
    unexpected = [name for name in extras if name not in declared]
    if unexpected:
        raise TypeError(f"launchOperator: undeclared extra(s) passed: {unexpected}")
    for name, spec_extra in declared.items():
        value = extras[name]
        is_tensor = isinstance(value, torch.Tensor)
        if spec_extra.kind is ExtraKind.TENSOR and not is_tensor:
            raise TypeError(f"launchOperator: extra {name!r} declared TENSOR, got {type(value)}")
        if spec_extra.kind is ExtraKind.SCALAR and is_tensor:
            raise TypeError(f"launchOperator: extra {name!r} declared SCALAR, got a torch.Tensor")

    additionalArguments = tuple(extras[e.name] for e in spec.extras)

    outputSizes = []
    outputDtypes = []
    for out in spec.outputs:
        shape, dtype = out.resolve(ctx, extras)
        outputSizes.append(shape)
        outputDtypes.append(dtype)
    if len(spec.outputs) == 1:
        outputSizes, outputDtypes = outputSizes[0], outputDtypes[0]

    jvp_fn = _build_geometry_jvp_fn(spec, ctx, extras) if spec.jvp is not None else None

    queryVolumes, referenceVolumes = ctx.corrections.volumes

    defaultStateArguments = (
        ctx.query, ctx.properties, ctx.domain,
        queryVolumes, referenceVolumes,
        ctx.adjacency,
        ctx.reference,
        ctx.corrections.crk,
        ctx.corrections.gradH,
        ctx.corrections.renorm,
    )

    return _launch(
        launcher=launch_kernel,
        kernel=spec.kernel,
        outputSizes=outputSizes,
        outputDtypes=outputDtypes,
        defaultStateArguments=defaultStateArguments,
        additionalArguments=additionalArguments,
        numThreads=spec.numThreads(ctx, extras) if spec.numThreads is not None else None,
        jvp_fn=jvp_fn,
    )
