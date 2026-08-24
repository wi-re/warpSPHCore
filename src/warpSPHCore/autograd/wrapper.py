import torch
import torch.autograd.forward_ad as fwAD
import warp as wp
from ..util import *
from .cache import *
from typing import Callable, Optional
from .stateAwareWarpFunction import StateAwareWarpFunction
from .arg_extract import extractStateInfo


def _isInertTensor(t: torch.Tensor) -> bool:
    """warpier_jvp_dual_argument_pruning_plan.md Phase 1/2 spike: True iff
    *t* carries no forward-mode tangent right now (``unpack_dual`` -- pure
    metadata inspection, no GPU sync, ~15-100x cheaper than the
    ``.abs().max()>0`` sync ``hasLiveTangent`` needs) and is not
    ``requires_grad`` -- i.e. this specific call has no gradient information
    to push through *t* in either direction, forward or reverse. A per-call,
    per-tensor runtime fact, never a static per-field schema (the "constant()
    means never-differentiable" direction was proposed and explicitly
    rejected -- see the plan)."""
    _, tangent = fwAD.unpack_dual(t)
    return tangent is None and not t.requires_grad


def _launch(
    launcher,
    kernel,
    outputSizes,
    outputDtypes,
    defaultStateArguments: tuple,
    additionalArguments: tuple = (),
    numThreads: Optional[int] = None,
    jvp_fn: Optional[Callable] = None,
):
    """Shared engine behind both ``warpWrapper2`` (the untyped, positional
    legacy entry point) and ``launchOperator`` (Step I, ``operator_spec.py``
    -- the declared-ABI entry point that resolves an ``OperatorSpec`` /
    ``SPHContext`` into exactly this call shape). Extracted so the two
    surfaces cannot drift: same struct assembly, same autograd bridge, same
    behaviour, differing only in how a caller arrives at these arguments.
    See warpier_fields.md Section 8 / Step I.

    ``jvp_fn`` (warpier_unified_operator_wrapper_plan.md): an already-built
    ``StateAwareWarpFunction.jvp()`` delegate, or ``None``.
    ``launchOperator`` builds this (``_build_geometry_jvp_fn``, Phase 2 --
    supersedes Phase 1's flat-position-only value-tangent closure, which
    delegated to ``warpOperationJVP``'s own value-tangent fallback path
    anyway) from ``OperatorSpec.jvp`` and this call's ``SPHContext`` before
    calling ``_launch``; it needs no further translation here since it is
    built with the flat-tensor layout *this* call will produce already in
    mind. ``None`` (the default, and always for ``warpWrapper2``, which has
    no ``OperatorSpec``) means no JVP support: passed straight through to
    ``StateAwareWarpFunction.apply``, unaffected.
    """
    with record_function("warpWrapper2 [WW2]"):
        # --- extract state tensors and the struct-building closure ---
        flat_state_tensors, state_build_fn, device, dim = extractStateInfo(
            *defaultStateArguments
        )
        n_state = len(flat_state_tensors)

        # --- split additionalArguments into tensors and non-tensors ---
        add_tensor_pos = []   # (original_index, tensor)
        add_scalar_map = {}   # original_index -> scalar value
        for i, arg in enumerate(additionalArguments):
            if isinstance(arg, torch.Tensor):
                add_tensor_pos.append((i, arg))
            else:
                add_scalar_map[i] = arg

        add_tensors = [t for _, t in add_tensor_pos]
        n_add       = len(additionalArguments)

        # --- unified flat tensor list (state first, then additional tensors) ---
        flat_tensors = flat_state_tensors + add_tensors
        n_total = len(flat_tensors)

        # --- prune inert real Tensors out of apply()'s tracked-argument list
        # (warpier_jvp_dual_argument_pruning_plan.md). Once ANY argument to a
        # torch.autograd.Function.apply() call is a dual tensor, torch's
        # forward-mode-AD dispatch synthesizes a fresh zero-filled tangent
        # for every OTHER real-Tensor argument in that same call -- the
        # dominant remaining jvp/fd overhead after this plan's Fixes 1+2. A
        # tensor that is neither dual nor requires_grad right now carries no
        # gradient information for this call in either direction, so it
        # never needs to reach StateAwareWarpFunction.apply() as a tracked
        # positional Tensor at all: convert it to its wp.array view here,
        # outside that boundary, and merge it back into the full-length
        # argument list inside build_fn/the jvp_fn wrapper below. Field
        # entries (not torch.Tensor) already bypass synthesis today and are
        # left exactly where they were -- this only prunes real Tensors.
        tracked_positions = []
        precomputed_wa = {}
        for i, t in enumerate(flat_tensors):
            if isinstance(t, torch.Tensor) and _isInertTensor(t):
                wa = getCachedWarpArray(t, use_cache=True)
                wa.requires_grad = False
                precomputed_wa[i] = wa
                continue
            tracked_positions.append(i)

        flat_tensors = [flat_tensors[i] for i in tracked_positions]

        # --- build_fn combines struct args + reconstructed additional args ---
        def build_fn(wa: list, use_bundle: bool = False) -> tuple:
            wa_full = [precomputed_wa.get(i) for i in range(n_total)]
            for pos, orig in enumerate(tracked_positions):
                wa_full[orig] = wa[pos]

            struct_args = state_build_fn(wa_full[:n_state], use_bundle=use_bundle)

            # Reconstruct additional args preserving original order
            reconstructed = [None] * n_add
            for pos, (orig_idx, _) in enumerate(add_tensor_pos):
                reconstructed[orig_idx] = wa_full[n_state + pos]
            for orig_idx, val in add_scalar_map.items():
                reconstructed[orig_idx] = val

            return struct_args + tuple(reconstructed)

        # Wrap launcher to inject numThreads if provided
        if numThreads is not None:
            original_launcher = launcher
            def launcher_with_threads(kernel, output_shape, output_dtype, *args):
                return original_launcher(kernel, output_shape, output_dtype, *args, numThreads=numThreads)
            launcher = launcher_with_threads

        # jvp_fn (operator_spec.py's _build_geometry_jvp_fn) indexes
        # flat_tangents by fixed absolute position (_QPOS, _RPOS, ... --
        # extractStateInfo's documented 36-slot layout), unaware of this
        # call's pruning. Wrap it to expand the pruned flat_tangents back
        # into a full n_total-length list at those same absolute positions,
        # with None at every pruned (by construction: no live tangent)
        # position -- so operator_spec.py needs no changes at all. Only
        # wrap when pruning actually happened this call (the common case
        # once anything is genuinely dual: most structural arguments prune).
        if jvp_fn is not None and len(tracked_positions) != n_total:
            _orig_jvp_fn = jvp_fn

            def jvp_fn(autogradCtx, flat_tangents_pruned):
                full_tangents = [None] * n_total
                for pos, orig in enumerate(tracked_positions):
                    full_tangents[orig] = flat_tangents_pruned[pos]
                pruned_live_mask = getattr(autogradCtx, "_jvp_live_mask", None)
                if pruned_live_mask is not None:
                    full_live_mask = [False] * n_total
                    for pos, orig in enumerate(tracked_positions):
                        full_live_mask[orig] = pruned_live_mask[pos]
                    autogradCtx._jvp_live_mask = full_live_mask
                return _orig_jvp_fn(autogradCtx, full_tangents)

        return StateAwareWarpFunction.apply(
            jvp_fn, build_fn, launcher, kernel, outputSizes, outputDtypes,
            *flat_tensors,
        )


def warpWrapper2(
    launcher,
    kernel,
    outputSizes,
    outputDtypes,
    defaultStateArguments: tuple,
    additionalArguments: tuple = (),
    numThreads: Optional[int] = None,
):
    """
    State-aware autograd wrapper for SPH kernels -- the untyped, positional
    legacy entry point (Section 8.1: a None-padded 10-tuple, re-analysed
    ``additionalArguments`` every call). Kept as a thin shim over ``_launch``
    for existing call sites; new code should declare an ``OperatorSpec`` and
    call ``launchOperator`` instead (``operator_spec.py``, Step I).

    Args:
        launcher:               Kernel launcher, e.g. ``launch_kernel``.
        kernel:                 The ``wp.kernel`` to execute.
        outputSizes:            Output shape passed to the launcher.
        outputDtypes:           Output Warp dtype(s) passed to the launcher.
        defaultStateArguments:  Tuple in the fixed state-argument order:
                                    (queryParticles, operationProperties, domain,
                                     queryVolumes, referenceVolumes, adjacency,
                                     referenceParticles, crkState,
                                     gradHState, renormalizationState)
        additionalArguments:    Extra per-kernel arguments appended after the
                                standard struct args.  Any ``torch.Tensor`` entries
                                will be tracked for gradients; plain Python scalars
                                and ints are forwarded unchanged.
        numThreads:             Explicit thread count for wp.launch(). If None,
                                defaults to outputSizes. Use this when the number
                                of threads should differ from output size.

    Returns:
        torch.Tensor or tuple of torch.Tensor – kernel output(s).
    """
    return _launch(
        launcher, kernel, outputSizes, outputDtypes,
        defaultStateArguments, additionalArguments, numThreads,
    )
