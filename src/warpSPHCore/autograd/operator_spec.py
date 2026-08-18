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
    DomainDescription,
    ExecutionMode,
    GradHState,
    OperationProperties,
    ParticleState,
    RenormalizationState,
)
from .launcher import launch_kernel
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
class OperatorSpec:
    """Declared once per kernel, at import time, next to the kernel itself.
    See warpier_fields.md Section 8.2.

    ``numThreads``, when set, is a resolver called as ``(ctx, extras) ->
    int`` and passed through to ``_launch`` explicitly -- for the rare
    kernel whose thread count is not the shape of its first output (see
    ``ThreadSpec``'s docstring). ``None`` (the default) preserves today's
    behaviour: ``_launch`` derives it from the first output's shape.
    """

    kernel: Any
    outputs: Tuple[OutputSpec, ...]
    extras: Tuple[ExtraSpec, ...] = ()
    threads: ThreadSpec = ThreadSpec.QUERY_COUNT
    numThreads: Optional[Callable[["SPHContext", Dict[str, Any]], int]] = None


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
    )
