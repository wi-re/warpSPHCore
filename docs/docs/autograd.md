# Autodiff machinery

How `torch.autograd` sees Warp kernels. Module:
`src/warpSPHCore/autograd/`.

A Warp kernel launch is opaque to PyTorch: the kernel reads `wp.array`
views of torch tensors and writes into a Warp-allocated buffer, so
neither `torch.autograd.Function`'s default bookkeeping nor generic
torch composition of "Warp-backed" callables works — a `wp.Tape`
records the exact `wp.array` *objects* it was launched with, and its
`backward()` re-reads mutable struct fields lazily (verified against
warp 1.16.0, see [caching layers](#caching-layers)). The bridge is
`StateAwareWarpFunction(torch.autograd.Function)` in
`autograd/stateAwareWarpFunction.py`:

- **forward** converts the torch inputs to `wp.array` views, runs the
  kernel under a `wp.Tape` when anything requires grad, and returns
  torch tensors;
- **backward** seeds the tape's gradients with
  `tape.backward(grads={output_wp_array: grad})` and reads each input
  array's `.grad` back;
- **jvp** delegates to the operator's own
  [`warpOperationJVP`](#the-36-slot-flat-layout) — the same
  already-exhaustively-tested dispatch/gating logic, not a re-derivation.

## Call chain

```
warpOperation(...)                       # operations.py — validation,
  └─ _computeSPH<Op>_stateBackend        #     flattening, output reshape
       └─ launchOperator(spec, ctx, …)   # autograd/operator_spec.py —
           or warpWrapper2(...)          #     the legacy positional shim
            └─ _launch                   # autograd/wrapper.py — the shared engine
                 ├─ extractStateInfo     # autograd/arg_extract.py — flat tensors
                 │                        #   + build_fn closure
                 ├─ (prune inert tensors)
                 └─ StateAwareWarpFunction.apply(jvp_fn, build_fn,
                        launcher=launch_kernel, kernel, shape, dtype, *flat_tensors)
                      └─ launch_kernel   # autograd/launcher.py — torch-allocated
                           #                output, wp.from_torch view, wp.launch
```

`launchOperator` and `warpWrapper2` resolve into the *same* `_launch`
engine — they differ only in how a caller arrives at the arguments
(declared spec vs. None-padded positional tuples).

## The 36-slot flat layout

`extractStateInfo(queryParticles, operationProperties, domain,
queryVolumes=None, referenceVolumes=None, adjacency=None,
referenceParticles=None, crkState=None, gradHState=None,
renormalizationState=None)` flattens every tensor embedded in the
structured state arguments into one **deterministically-ordered list of
36 entries**, and returns a `build_fn` closure that reconstructs all
Warp kernel structs (particle states, `kernelState`, domain,
adjacency/grid, corrections) from a parallel list of `wp.array`s:

| slot | field | slot | field | slot | field |
|---|---|---|---|---|---|
| 0 | `qPos` | 12 | `rOmega` (grad-h) | 24 | `adj_neighborOffsets` |
| 1 | `rPos` | 13 | `qVol` | 25 | `adj_numNeighbors` |
| 2 | `qSup` | 14 | `rVol` | 26 | `grid_sortIndex` |
| 3 | `rSup` | 15–18 | `qcrk_A/B/gradA/gradB` | 27 | `grid_qMin` |
| 4 | `qMas` | 19–22 | `rcrk_A/B/gradA/gradB` | 28 | `grid_qMax` |
| 5 | `rMas` | 23 | `adj_neighborList` | 29 | `grid_numCells` |
| 6 | `qDen` | 30 | `grid_hashTable` | 33 | `domainMin` |
| 7 | `rDen` | 31 | `grid_cellTable` | 34 | `domainMax` |
| 8 | `qK` (kinds) | 32 | `grid_cellOffsets` | 35 | `periodicity` |
| 9 | `rK` | 10 | `renormMat` | 11 | `qOmega` |

The list is **heterogeneous**: real inputs are `torch.Tensor`s, but
disabled correction slots (no CRK, no grad-h, no volumes, no
renormalization, the unused half of adjacency-vs-grid) are filled with
a permanent `Field` placeholder from the null-field registry
(`util/fieldRegistry.nullField`) — a standalone, never-differentiable
view built once, never re-converted. `ParticleState.kinds` is required
(no more `hasattr` probe or `None` fallback). `referenceParticles=None`
aliases the query tensors at the reference slots (qPos *is* rPos, the
same object) — which is exactly the aliased-role case
[backward](#backward) has to deduplicate.

The [lattice-normalization
calibration](renorm#lattice-normalization-calibration) is resolved here
too: when `calibrateNormalization` is set,
`latticeDensityFactor(kernel, n_h, dim)` becomes
`normalizationCoefficient` in the config dict (raising if not finite
and positive) and is carried into the Warp `kernelState` struct.

The JVP machinery relies on this layout being **fixed**: it reads
tangent positions by absolute index (`_QPOS=0, …, _RENORM_MAT=10,
_QVOL/_RVOL=13/14, _QCRK_*=15..18`).

## The bridge: `StateAwareWarpFunction`

`apply(jvp_fn, build_fn, launcher, kernel, output_shape, output_dtype,
*flat_tensors)` — the first six are non-tensor (6 `None`s must be
returned in backward before the per-tensor gradients).

### Forward

1. **Save.** `save_for_backward` only the real-`Tensor` positions
   (gated on `any_requires_grad`); `save_for_forward` the same set, but
   only when a `jvp_fn` is registered — the documented separate store
   `jvp()` may read via `ctx.saved_tensors`.
2. **Convert.** Each tensor goes through
   `getCachedWarpArray(t, use_cache=…)`:
   - the **no-grad path always caches** — the view is attached to the
     tensor *object* via the Field registry (`acquireView`), so a
     repeated call on the same tensor reuses the `wp.array` wrapper
     (and its `wp.from_torch` zero-copy view) instead of rebuilding;
   - the **grad path caches too, gated by
     `WARPSPHCORE_FIELD_CACHE_GRAD`** (default on; set `"0"` to bisect a
     suspected caching bug without a full revert);
   - `wa.requires_grad = t.requires_grad` is preserved so the tape
     tracks it, and — the part that makes wrapper reuse safe —
     **zero-on-acquire**: `wa.grad.zero_()` whenever the tensor requires
     grad. `wp.Tape.get_adjoint` returns `wa.grad` directly (Warp's own
     persistent per-array buffer, not a fresh per-tape one), so a cached
     wrapper's `.grad` is genuinely shared object state across every
     call that acquires it. The contract "the gradient buffer is zero at
     the start of forward" is enforced at the point of use rather than
     relying on some earlier `tape.zero()` having run (the earlier
     call's output might never have been used in a `.backward()`).
   - `t` itself is passed (not `t.detach()`): the registry attaches to
     the object it is given, and `detach()` returns a fresh object every
     call, so caching against a pre-detached tensor would never hit.
3. **Build.** `build_fn(warp_arrays, use_bundle=…)` reconstructs the
   struct arguments. `use_bundle=True` (the `StateBundle` struct reuse,
   [below](#caching-layers)) **only when nothing requires grad** —
   unconditionally, with no zero-on-acquire equivalent that would make
   sharing safe (see the bundle docstring for the repro).
4. **Launch.** Under a `wp.Tape` if `any_requires_grad`, else bare.
   The output is allocated on **torch's** caching allocator
   (`launch_kernel`/`_allocate_output` — one allocator, not two
   independently-growing GPU pools); the `wp.array` view the tape
   recorded is stashed as `ctx.output_warp`, and the `_warp_array`
   courier attribute is deleted immediately (it would otherwise close a
   tensor↔wp.array reference cycle only the cyclic GC can break — a
   sawtooth allocation pattern on every launch).

### Backward

`tape.backward(grads={output_wp_array: grad_output.contiguous()})` —
seeding the tape's gradient dictionary rather than assigning
`array.grad` directly is what makes the tape's own kernels accumulate
into the right buffer. Then each distinct input `wp.array`'s `.grad`
is read back **exactly once**:

> An input tensor that fills two roles in one call (the common case:
> `referenceParticles=None` ⇒ qPos and rPos are literally the same
> object) maps to the *same* cached `wp.array` at both flat positions.
> Warp's adjoint kernels have already accumulated *both* roles'
> contributions into that one shared `.grad` buffer, so reading it
> again for the second role would report the complete total a second
> time — and PyTorch sums whatever `backward` returns across every
> position a leaf occupies, **silently doubling the gradient**. The
> `seen_wa_ids` set makes later aliases report `None`. (With caching
> off, aliased positions are distinct objects with independent
> partial-`.grad` buffers and the dedup is a no-op.)

Finally `tape.zero()` clears the tape.

### JVP

`jvp(ctx, *tangents)` — torch guarantees the tangents align
positionally with `forward`'s arguments. Two subtleties:

- **Torch synthesizes zeros, not `None`s.** Once *any* argument of an
  `apply()` call is a dual tensor, torch's forward-mode dispatch
  synthesizes a **zero tensor** as the tangent for every *other*
  real-tensor argument in the same call (only genuinely non-tensor
  arguments — closures, `Field` placeholders — come back as `None`).
  "Nothing is being differentiated" must therefore be checked as
  *none-of-them-live*, where live means `abs().max() > 0`.
- **One sync, not one per tensor.** `_liveTangentMask` reduces every
  non-`None` tensor's `.abs().max()` on-device, stacks the results, and
  pays exactly **one** GPU→CPU round trip (`.tolist()`) for the whole
  call. Profiled on the wave-equation JFNK(jvp) benchmark
  (2026-08-24): the per-tensor version was ~16 % of wall-clock, ~95 %
  of the checks finding nothing live.

If no position is live, `jvp()` returns `None` (no tangent output — the
dual wrapper sees a plain value). If something *is* live but this
kernel has no `JVPSpec` registered (`jvp_fn is None`), it **raises**
rather than silently returning a tangent-free dual output. The mask is
stashed on the ctx and handed to `jvp_fn`, which re-consults the same
positions many times over without re-paying the sync.

## `_launch` and inert-tensor pruning

Before calling `apply`, `_launch` (autograd/wrapper.py) prunes
**inert** tensors out of the tracked-argument list:
`_isInertTensor(t)` is true iff `fwAD.unpack_dual(t)` shows no forward
tangent *right now* (pure metadata inspection — no GPU sync, 15–100×
cheaper than the `.abs().max()` sync) **and** `not t.requires_grad` —
i.e. this call has no gradient information to push through `t` in
either direction. An inert tensor is converted to its `wp.array` view
*outside* `apply`'s boundary (cached) and merged back into the
full-length argument list inside `build_fn`; when a `jvp_fn` exists, a
wrapper re-expands the pruned `flat_tangents` back to full length (with
`None` at every pruned position — inert means, by construction, no live
tangent). This stops torch's zero-tangent synthesis from touching
dozens of structural arguments on every dual call — the dominant
remaining JVP/fd overhead after the value-tangent fixes. It is a
per-call, per-tensor runtime fact, never a static per-field schema.

## The declared ABI: `OperatorSpec` / `SPHContext` / `launchOperator`

`autograd/operator_spec.py` formalizes what every `warpWrapper2` call
site already does by convention (a fixed 10-slot positional state tuple,
None-padded per call; an `additionalArguments` tuple re-analysed by
`isinstance` every launch; an output dtype/shape probed inline):

```python
@dataclass(frozen=True)
class OutputSpec:    # dtype and/or shape may be a resolver (ctx, extras) -> value
                     #   (needed when the output packs a runtime field shape)
@dataclass(frozen=True)
class ExtraSpec:     # one declared additionalArguments name, kind TENSOR|SCALAR
@dataclass(frozen=True)
class OperatorSpec:
    kernel: Any
    outputs: Tuple[OutputSpec, ...]
    extras: Tuple[ExtraSpec, ...] = ()
    threads: ThreadSpec = ThreadSpec.QUERY_COUNT   # or a numThreads resolver
    jvp: Optional[JVPSpec] = None

@dataclass
class SPHContext:    # per-call, named (non-positional) state
    query: ParticleState
    properties: OperationProperties
    domain: DomainDescription
    adjacency=None; reference=None
    corrections: Corrections   # volumes (q, r), crk, gradH, renorm
    mode: ExecutionMode = AUTO # FORWARD is rejected, deliberately unimplemented
```

`launchOperator(spec, ctx, **extras)` validates the extras against the
declaration (missing/undeclared/type-mismatch all raise), resolves the
outputs, and — when `spec.jvp` is set — builds the `jvp_fn` closure,
then delegates to `_launch`. **It is not a new caching layer and not a
behaviour change**: a kernel ported to the ABI launches identically to
the `warpWrapper2` call it replaces.

`JVPSpec(queryValueExtra, referenceValueExtra)` names the declared
`extras` entries eligible for a Tier-1 *value* tangent:
Interpolate — `referenceValueExtra="referenceValues"` only (its kernel
never reads `queryValues`); Gradient/Divergence/Curl/Laplacian —
`"queryValuesFlat"`/`"referenceValuesFlat"`; Density/Covariance — both
`None` (geometry tangents only, no value input).

### The JVP closure

`_build_geometry_jvp_fn(spec, ctx, extras)` covers the Tier-1 value
tangent, the Tier-2 geometry tangent, and their *sum* (both live) in
one path by **delegating entirely to `warpOperationJVP`** — not
re-deriving any of its dispatch/gating logic (the CRK/renormalization
scope tables, the Laplacian-scheme restrictions, the
Divergence/Curl restrictions). It captures this call's `SPHContext` by
closure, so every *primal* semantic object (`ParticleState`,
`CRKState`, `RenormalizationState`, `adjacency`, `domain`, the primal
value tensors) is the caller's own original Python object — never
reconstructed from flat tensors. Only the *tangent* objects are
rebuilt from `flat_tangents`'s fixed positions:

- `ParticleTangentState` for query/reference (positions/supports/masses/
  densities), a field taken iff live, else `None`;
- `CRKTangentState` when this call carries CRK and any of the four
  slots is live — its dataclass has **no `Optional` fields**, so a
  primal-shaped zero is substituted for whichever of A/B/gradA/gradB
  has no live tangent (a frozen correction);
- `RenormalizationTangentState` when the renorm-matrix slot is live;
- value tangents via `_unflattenValue`: the `…ValuesFlat` extras are
  `view(-1, flatInputShape)` of the caller's original field, and the
  `numDims` scalar extra inverts the scalar-field case (`[N]` →
  `[N, 1]`); rank ≥ 2 fields **raise `NotImplementedError`** (the
  flattened shape cannot be inverted without the original per-dimension
  shape — call `warpOperationJVP` directly);
- when only one side of a two-sided value tangent is live, the other
  side gets `zeros_like` (its contribution is exactly zero);
- a live tangent on a **domain-bound or grid-traversal argument**
  (flat positions 27, 28, 33, 34 — `_NO_TANGENT_HOME`) **raises**: no
  operator's geometry JVP differentiates w.r.t. domain bounds or grid
  structure, so this is an explicit guard, not a silent drop;
- `_reflattenResult`: `forward` returns the kernel's *raw* output
  `[N, flatOutputShape]` while `warpOperationJVP` returns the
  public (already-reshaped) one, so the closure reshapes back via the
  `flatOutputShape` extra — a no-op for Interpolate/Density/Covariance,
  load-bearing for Laplacian and Divergence, whose public shape
  collapses a dimension the flat form still carries;
- multi-output kernels (Covariance: matrix + neighbor count) return
  `(tangent,) + (None,) * (n-1)` — `jvp()` must return one entry per
  `forward` output.

## Caching layers

Three deliberately-separated layers, all keyed to *tensor identity*,
never to storage address:

| layer | module | gate | what it reuses |
|---|---|---|---|
| `wp.array` view cache | `autograd/cache.py` (`getCachedWarpArray(use_cache=True)` → `util/fieldRegistry.acquireView`) | always on for no-grad; `WARPSPHCORE_FIELD_CACHE_GRAD` (default on) for grad | the `wp.from_torch` zero-copy view of a tensor *object* |
| `StateBundle` struct reuse | `util/stateBundle.py` | **no-grad only** (unconditional); `WARPSPHCORE_DISABLE_BUNDLE=1` is a *measurement* hatch, not a correctness one | the persistent Warp struct instances `build_fn` would otherwise rebuild every launch; refreshed in place |
| dummy-tensor cache | `autograd/cache.py` | always | tiny shared placeholders (null fields, identity matrices) |

The history matters: an earlier version keyed `wp.array` wrappers by
`(data_ptr, shape, strides, dtype)`. Reused wrappers share their
`.grad` buffer, which was never zeroed between unrelated calls — two
calls sharing storage (any training loop, `torch.autograd.gradcheck`)
silently accumulated gradients from prior calls, making backward
non-reentrant. The cache was **removed** rather than reasoned about;
its reintroduction is what the Field registry + zero-on-acquire above
are.

And the `StateBundle` no-grad gate is a **correctness requirement, not
a nicety**: `wp.Tape` does not snapshot a struct's field values at
launch — it holds a live reference to the (mutable) struct and
re-reads its fields lazily at `tape.backward()` time (verified with a
minimal two-launch repro against warp 1.16.0: the second call's array
ended up in the first call's gradient). A grad-requiring call therefore
always gets fresh, call-local structs.

## HVP

`warpOperationHVP` (the Density position Hessian) is **not** a generic
torch composition of two JVPs — generic composition of Warp-backed
functions does not work through the bridge (the tape cannot see through
the `Function` boundary the way a torch-native HVP recipe needs). It is
its own dedicated Warp kernel (with the exact self-pair term — see the
[Density](operations/density) page), pinned by
`tests/operations/test_forward_mode_tier2_density_hvp_self_pair.py`.

## The no-host-sync invariant

The forward path of an operator launch (extract → convert → build →
launch) performs **no GPU→CPU synchronization** on the CUDA path —
the one sync in the whole machinery is `jvp()`'s batched liveness mask,
and only when forward-mode AD is actually in use. This is a standing
invariant with a census, not a hope:
`tests/operations/test_no_host_sync.py` counts readbacks on specific
modules, `scripts/count_host_syncs.py` does the whole-repo census, and
the motivating before/after records are in `docs/regression/` (the
[renormalization](renorm) fallback is the canonical example).

## Tests

- `tests/operations/test_gradcheck_scripts.py` — gates the
  `scripts/gradcheck_*.py` float64 `torch.autograd.gradcheck` suites
  (reverse mode through the full bridge, every operator × correction
  combination).
- `tests/operations/test_forward_mode_value_jvp.py`,
  `test_forward_mode_geometry_jvp_*.py` — the JVP paths, including
  through the dual-tensor bridge.
- `tests/operations/test_forward_mode_dual_wrapper.py` — the bridge's
  `jvp()` dispatch itself (live-mask, raise-on-unregistered, zeros).
- `tests/operations/test_field_abstraction.py`,
  `test_null_field_wiring.py` — the Field/null-field registry
  semantics.
- `tests/operations/test_state_bundle.py`, `test_view_reuse.py` — the
  two caching layers and their gates.
- `tests/operations/test_no_host_sync.py` — the sync invariant.

## See also

[Operator pages — the JVP formulas the bridge dispatches to](operations/density)
· [Renormalization & lattice calibration](renorm) (resolved in
`extractStateInfo`) · `warpier_adjoint.md` (the JVP derivations) ·
`warpier_fields.md` (the state-object design record the caching layers
implement)
