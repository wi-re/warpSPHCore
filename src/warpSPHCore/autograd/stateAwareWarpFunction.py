import os

import torch
import warp as wp
from ..util import *
from .cache import *
from ..profiling import record_function


def _field_cache_grad_enabled() -> bool:
    # warpier_fields.md Step E: view reuse defaults to *on* for the grad path
    # too, once landed -- set to "0" to force fresh builds on the grad path
    # only (the no-grad path is unconditionally cached regardless of this
    # variable) for bisecting a suspected caching bug without a full revert.
    return os.environ.get("WARPSPHCORE_FIELD_CACHE_GRAD", "1") != "0"


def _bundle_enabled() -> bool:
    # warpier_fields.md Step H: a *measurement* hatch, not a correctness one.
    # Step F deliberately ships no env switch for StateBundle reuse -- the grad
    # path never shares a bundle (there is no contract that would make that safe,
    # so the gate is `requires_grad`, not a variable), which leaves nothing to
    # bisect. But Step H's in-situ before/after needs to turn every Steps A-F
    # caching layer off on the *no-grad* path to measure what they bought on a
    # real workload, and without this the bundle stayed on in both arms and the
    # comparison silently understated the win. Turning it off only ever falls
    # back to arg_extract.py's original per-call struct construction, so this can
    # never be the unsafe direction.
    return os.environ.get("WARPSPHCORE_DISABLE_BUNDLE", "0") != "1"


class StateAwareWarpFunction(torch.autograd.Function):
    """
    Autograd bridge for SPH operations called with structured state arguments
    (ParticleState, CRKState, etc.).

    The caller supplies a *build_fn* closure that maps a flat list of Warp arrays
    (one per tracked torch.Tensor in the flat_tensors list) to the full argument
    tuple expected by the launcher.  This keeps all struct-assembly logic out of
    the autograd Function while still letting gradients flow through every tensor
    that had requires_grad=True.

    Forward signature (excluding ctx):
        jvp_fn          - closure: (ctx, flat_tangents) -> tangent output(s), or
                          None (warpier_unified_operator_wrapper_plan.md Phase 1:
                          built by `_launch` from `OperatorSpec.jvp`; None means
                          this call site has no JVP support registered -- a live
                          tangent reaching `jvp()` then raises rather than
                          silently dropping it, see `jvp()` below)
        build_fn        - closure: List[wp.array] -> tuple of kernel args
        launcher        - e.g. launch_kernel
        kernel          - the wp.kernel to execute
        output_shape    - scalar int or list[int] for multi-output
        output_dtype    - warp dtype or list[warp dtype] for multi-output
        *flat_tensors   - torch.Tensor or Field entries, in deterministic order
                          (warpier_fields.md Step C: Field entries are the
                          null-field registry's standalone, never-differentiable
                          placeholders -- see the forward()/backward() split
                          below for why only the Tensor entries participate in
                          save_for_backward)
    """

    # Number of non-tensor leading arguments (jvp_fn, build_fn, launcher,
    # kernel, output_shape, output_dtype).  backward() must return this many
    # Nones before the per-tensor gradients.
    _N_NON_TENSOR = 6

    @staticmethod
    def forward(ctx, jvp_fn, build_fn, launcher, kernel, output_shape, output_dtype, *flat_tensors):
        # with record_function(f"Warp Function State Aware Forward"):
        ctx.jvp_fn = jvp_fn
        ctx.build_fn = build_fn
        ctx.launcher = launcher
        ctx.kernel = kernel
        ctx.output_shape = output_shape
        ctx.output_dtype = output_dtype
        # jvp()'s own reconstruction needs the *original* (Tensor-or-Field)
        # flat_tensors list positionally, not just the converted wp arrays
        # below -- Field entries pass straight through a nested relaunch
        # unchanged, and save_for_forward (below) only accepts real Tensors.
        ctx.flat_tensors_raw = flat_tensors

        # flat_tensors is heterogeneous from Step C on: Field entries are the
        # null-field registry's standalone, permanent placeholders (never
        # differentiable), Tensor entries are the caller's real inputs.
        # save_for_backward only accepts actual tensors, so it -- and the
        # backward-side zip against it -- must track only the Tensor
        # positions; Field positions always contribute a None gradient.
        is_tensor = tuple(isinstance(t, torch.Tensor) for t in flat_tensors)
        ctx.is_tensor = is_tensor

        ctx.any_requires_grad = any(
            t.requires_grad for t, tt in zip(flat_tensors, is_tensor) if tt
        )
        if ctx.any_requires_grad:
            ctx.save_for_backward(*(t for t, tt in zip(flat_tensors, is_tensor) if tt))
        else:
            # Avoid save_for_backward overhead when gradients are not requested.
            ctx.save_for_backward()

        # save_for_forward: PyTorch's documented mechanism for what jvp() may
        # read back via ctx.saved_tensors -- a *separate* store from
        # save_for_backward above (confirmed empirically, Phase 0 spike: the
        # two coexist on the same ctx without conflict, each resolving to the
        # right set of tensors in backward() vs. jvp() respectively). Gated on
        # jvp_fn being set, mirroring save_for_backward's own
        # any_requires_grad gate above -- no forward-mode caller, no reason to
        # pay for it.
        if jvp_fn is not None:
            ctx.save_for_forward(*(t for t, tt in zip(flat_tensors, is_tensor) if tt))

        # Detach → warp, preserving requires_grad so the tape tracks them.
        # The no-grad path always caches (Step D). The grad path caches too
        # as of Step E, gated by WARPSPHCORE_FIELD_CACHE_GRAD so a suspected
        # caching bug can be bisected without a full revert. Field entries
        # are the null registry's permanent view -- built once, never
        # reconverted, and never differentiable, unaffected by either gate.
        use_cached_views = (not ctx.any_requires_grad) or _field_cache_grad_enabled()
        with record_function("SAWF.forward - convert"):
            warp_arrays = []
            for t, tt in zip(flat_tensors, is_tensor):
                if tt:
                    # Pass t itself, not t.detach(): the Field-registry cache
                    # (use_cache=True) attaches to whatever object it is
                    # given, and detach() returns a fresh object every call,
                    # so caching against an already-detached tensor would
                    # never hit. getCachedWarpArray detaches internally.
                    wa = getCachedWarpArray(t, use_cache=use_cached_views)
                    wa.requires_grad = t.requires_grad
                    if t.requires_grad and wa.grad is not None:
                        # Zero-on-acquire (Section 3.3), unconditionally --
                        # this is the one thing that makes reusing a wrapper
                        # across calls safe. wa.grad is warp's own persistent
                        # per-array buffer (wp.Tape.get_adjoint returns it
                        # directly, not a fresh per-tape one), so once a
                        # wrapper is cached, its .grad is genuinely shared
                        # object state across every call that acquires it --
                        # exactly the state the original data_ptr-keyed
                        # cache reused without zeroing, silently accumulating
                        # gradients from one call into the next. The
                        # contract here is "the gradient buffer is zero at
                        # the start of forward", enforced at the point of
                        # use rather than relying on a tape.zero() that ran
                        # (or didn't) at the end of some earlier backward --
                        # e.g. because that earlier call's output was never
                        # actually used in a .backward().
                        wa.grad.zero_()
                else:
                    wa = t.view()
                    wa.requires_grad = False
                warp_arrays.append(wa)
            ctx.warp_arrays = warp_arrays

        # Reconstruct all kernel args via the caller-supplied closure.
        # use_bundle=True (warpier_fields.md Step F) only when nothing in
        # this call requires grad -- reusing a mutable struct across a
        # grad-requiring call would corrupt an earlier call's gradient the
        # moment its backward is deferred past a later refresh of the same
        # struct (wp.Tape reads struct fields lazily at backward() time, not
        # at launch time; see stateBundle.py). Same shape as use_cached_views
        # above and for the same reason, but this gate is unconditional --
        # unlike Step E's view-reuse cache, struct reuse has no zero-on-
        # acquire equivalent that would make grad-path sharing safe.
        with record_function("SAWF.forward - build_fn"):
            kernel_args = build_fn(
                warp_arrays,
                use_bundle=(not ctx.any_requires_grad) and _bundle_enabled(),
            )

        with record_function("SAWF.forward - launch"):
            if ctx.any_requires_grad:
                tape = wp.Tape()
                with tape:
                    output = launcher(kernel, output_shape, output_dtype, *kernel_args)
            else:
                tape = None
                output = launcher(kernel, output_shape, output_dtype, *kernel_args)

        ctx.tape = tape
        # launcher() now returns torch tensors allocated on torch's own
        # allocator; the wp.array view used for the actual (taped) kernel
        # launch is stashed on each tensor by launch_kernel, since backward
        # must seed gradients through that exact object -- see the comment
        # in launcher.py's _allocate_output. The stash is only a courier
        # across launch_kernel's return boundary: wp.from_torch() gives the
        # wp.array a `_tensor` back-reference to its source tensor, so
        # leaving `_warp_array` attached here closes a reference cycle
        # (tensor -> wp.array -> tensor) that only the cyclic GC -- not
        # refcounting -- can break, on its own schedule rather than as
        # memory is freed. Deleting it once read keeps the wp.array
        # reachable only via ctx.output_warp, a plain acyclic reference.
        if isinstance(output, (list, tuple)):
            ctx.output_warp = tuple(o._warp_array for o in output)
            for o in output:
                del o._warp_array
            return tuple(output)
        ctx.output_warp = output._warp_array
        del output._warp_array
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        N = StateAwareWarpFunction._N_NON_TENSOR
        n_flat = len(ctx.is_tensor)

        if not ctx.any_requires_grad:
            return (None,) * N + (None,) * n_flat

        # See WarpFunctionWrapper.backward's comment above for why this
        # seeds gradients via Tape.backward(grads=...) instead of assigning
        # array.grad directly.
        output_warp = ctx.output_warp
        grads = {}
        if isinstance(output_warp, (list, tuple)):
            for out, grad in zip(output_warp, grad_outputs):
                if grad is not None:
                    grads[out] = castTorchToWarpAsBuiltins(grad.contiguous())
        else:
            if grad_outputs[0] is not None:
                grads[output_warp] = castTorchToWarpAsBuiltins(grad_outputs[0].contiguous())

        ctx.tape.backward(grads=grads)

        # A source tensor that fills two roles in one call (the common case:
        # referenceParticles=None means qPos and rPos are literally the same
        # tensor object) now maps to the *same* cached wp.array at both flat
        # positions (warpier_fields.md Step E). Warp's adjoint kernels then
        # accumulate both roles' contributions into that one shared .grad
        # buffer already -- get_adjoint() returns `a.grad` directly, not a
        # fresh per-argument copy, so the buffer holds the complete summed
        # gradient the first time either role is read. Reading it again for
        # the second role would report that same complete total a second
        # time, and PyTorch sums whatever this returns across every position
        # a leaf tensor occupies, silently doubling the gradient. Read each
        # distinct wp.array's .grad exactly once; every later flat position
        # sharing it reports None so nothing it contributes gets double-
        # counted. (When the cache is off, aliased positions build distinct
        # wp.array objects with independent, correctly-partial .grad buffers
        # -- id(wa) is never shared then, so this is a no-op in that case.)
        seen_wa_ids: set[int] = set()
        saved = iter(ctx.saved_tensors)
        input_grads = []
        for wa, tt in zip(ctx.warp_arrays, ctx.is_tensor):
            if not tt:
                # Field entry (null-field registry): standalone, never
                # differentiable -- no corresponding saved tensor.
                input_grads.append(None)
                continue
            t = next(saved)
            if t.requires_grad and id(wa) not in seen_wa_ids:
                seen_wa_ids.add(id(wa))
                input_grads.append(wp.to_torch(wa.grad).clone())
            else:
                input_grads.append(None)
        ctx.tape.zero()  # Clear any accumulated gradients in the tape to avoid affecting future computations

        return (None,) * N + tuple(input_grads)

    @staticmethod
    def jvp(ctx, *tangents):
        # warpier_unified_operator_wrapper_plan.md Phase 1. tangents aligns
        # positionally with forward()'s (jvp_fn, build_fn, launcher, kernel,
        # output_shape, output_dtype, *flat_tensors) -- torch guarantees this
        # (confirmed against the installed torch 2.13 source in Phase 0).
        N = StateAwareWarpFunction._N_NON_TENSOR
        flat_tangents = tangents[N:]

        # Empirical finding from the Phase 0 spike (verified directly against
        # torch, not assumed from its docs): once ANY argument of this
        # apply() call is a dual tensor, torch's jvp() dispatch synthesizes a
        # ZERO tensor -- not None -- as the tangent for every OTHER
        # real-Tensor argument in the same call, even ones with
        # requires_grad=False that were never wrapped as dual at all. Only
        # genuinely non-Tensor forward() arguments (build_fn, launcher,
        # kernel, ..., Field placeholders) come back as None. So "no live
        # tangent anywhere in this call" must be checked as None-or-all-zero.
        # Compute the liveness mask ONCE and hand it to the jvp_fn closure
        # (which re-consults the same flat positions many times over -- see
        # _build_geometry_jvp_fn). Profiled on the wave-equation JFNK(jvp)
        # benchmark (2026-08-24): this GPU->CPU sync accounted for ~16% of
        # wall-clock time, ~95% of those checks finding nothing live -- most
        # positions in a typical call are exactly this zero-tangent filler
        # (e.g. position/support/mass/density arguments when only a value
        # field is actually being differentiated, as in that benchmark).
        # _liveTangentMask batches every tensor's own .abs().max() reduction
        # on-device (no sync each) and pays exactly ONE GPU->CPU round trip
        # for the whole call, instead of one per tensor.
        live_mask = _liveTangentMask(flat_tangents)
        if not any(live_mask):
            return None

        if ctx.jvp_fn is None:
            raise NotImplementedError(
                "StateAwareWarpFunction.jvp: a live (non-zero) tangent reached an "
                "operator launch with no JVPSpec registered (OperatorSpec.jvp is None "
                "for this kernel) -- raising rather than silently returning a "
                "tangent-free dual output. See warpier_unified_operator_wrapper_plan.md."
            )
        # ctx is per-apply() call, so stashing the mask here never leaks across
        # calls; jvp_fn reads it back via autogradCtx._jvp_live_mask.
        ctx._jvp_live_mask = live_mask
        return ctx.jvp_fn(ctx, flat_tangents)


def hasLiveTangent(t) -> bool:
    """None (no tensor / no tangent) and an all-zero synthesized filler
    (torch's own convention -- see `jvp()` above) both mean "nothing to
    propagate here"; only a tensor with at least one non-zero entry is a
    real tangent. Shared between `jvp()`'s own "anything live at all?" check
    and each `jvp_fn` closure's "is this a supported combination?" check
    (`wrapper.py`), so both apply the same definition of "live". Single-
    tensor primitive kept for that shared definition and for any direct
    caller; `jvp()` itself uses the batched `_liveTangentMask` below rather
    than calling this once per tensor (each call is its own GPU->CPU sync)."""
    return t is not None and bool(t.abs().max() > 0)


def _liveTangentMask(tangents) -> list:
    """Batched form of ``[hasLiveTangent(t) for t in tangents]``: reduces
    every non-None tensor's ``.abs().max()`` on-device (no sync per tensor --
    ``torch.Tensor.max()`` returns a GPU-resident 0-d tensor), stacks the
    results, and pays exactly ONE GPU->CPU round trip for the whole call via
    a single ``.tolist()`` instead of one round trip per tensor. Equivalent
    to (not an approximation of) the per-tensor version -- same "> 0" test,
    same None-stays-False handling -- just batched. See `jvp()`'s own comment
    for the profiled cost this replaces."""
    idx = [i for i, t in enumerate(tangents) if t is not None]
    mask = [False] * len(tangents)
    if not idx:
        return mask
    reduced = torch.stack([tangents[i].abs().max() for i in idx])
    for pos, isLive in zip(idx, (reduced > 0).tolist()):
        mask[pos] = isLive
    return mask


warpWrapperStateaware = StateAwareWarpFunction.apply
    