# Phase 3/4 Execution Plan: The `Field` Abstraction

Companion to `warpier_core.md`. That document defines Phase 3 ("Introduce a Field
Abstraction") and Phase 4 ("Improve State Construction") at the level of intent; this
one is the executable plan, written against the repo as it stands after `3cdd4e6`
("fix default allocation to be on torch...") and cross-checked against the `warpSPH`
frontend at `~/dev/warpSPH`.

Every number below was measured on this machine, not estimated. Bench harness:
`scripts/bench_call_overhead.py` (to be added as Step 0's deliverable; the ad-hoc
scripts it is derived from are in the session scratchpad).

---

# 1. The case for doing this now: measured baseline

Configuration: `N=20000`, 2D, float32, `WarpOperation.Density`, grid traversal,
`SupportScheme.Gather`, RTX 3090, warp 1.15.0, torch 2.13.0+cu130, CUDA 12.9.

## 1.1 The call is CPU-bound, and the CPU cost is N-independent

| N       | `warpOperation` wall time |
|---------|---------------------------|
| 2 000   | 919 us                    |
| 20 000  | 1009 us                   |
| 200 000 | 7484 us                   |

Between N=2k and N=20k the time barely moves. **Roughly 900 us per operator call is
fixed Python/marshalling overhead** that has nothing to do with problem size. Only at
N=200k does actual GPU work start to dominate. Every production case in the frontend
below ~100k particles is paying more for argument marshalling than for physics.

## 1.2 Where the 900 us goes

| Stage                                          | Time    | Share |
|------------------------------------------------|---------|-------|
| `extractStateInfo`                              | 69-92 us  | ~9%  |
| torch->warp conversion loop (36 tensors)        | 259-345 us | ~33% |
| `build_fn` (warp struct assembly)               | 56-71 us  | ~7%  |
| output allocation (`allocateTorchWarp`)         | 85 us   | ~9%   |
| `wp.launch` + `autograd.Function.apply` + rest  | ~350 us | ~40%  |

`cProfile` over 300 calls confirms the shape:

* `warp/_src/torch.py:from_torch` -- **38 calls per operator call**, 33% of cumulative time.
* `warp/_src/codegen.py:set_array_value` -- 36 calls per operator call.
* `warp/_src/codegen.py:__setattr__` (struct field writes) -- 51 calls per operator call.
* `torch.autograd.profiler` enter/exit -- 12 calls per operator call, and these
  `record_function` hooks are *unconditionally on* (94 of them across the package).

A single `castTorchToWarpAsBuiltins` costs 5.8-8.0 us. Thirty-eight of them per call
is the headline cost, and **almost all of them re-convert a tensor that was already
converted moments earlier.**

## 1.3 What the fix is worth (prototyped, not projected)

| Stage                    | Today   | Prototyped | Speedup |
|--------------------------|---------|------------|---------|
| conversion loop (36)     | 288 us  | 32.5 us    | 8.9x    |
| ... with attached fields | 288 us  | ~13 us     | ~22x    |
| `build_fn`               | 66.5 us | 1.6 us     | 41x     |

The 32.5 us figure is an `id()`-keyed dict cache with a full `(data_ptr, shape,
stride, dtype)` validity check. The ~13 us figure is the design actually recommended
below (field attached to the tensor): attribute lookup measures **0.060 us** against
**0.219 us** for a dict lookup keyed on `id()`, so the cheaper mechanism is also the
safer one.

**Target: ~900 us -> ~450-500 us of fixed overhead per operator call**, i.e. roughly
halving the CPU cost of every SPH operation in both repos.

## 1.4 The gradient path benefits too

| Per differentiable input tensor            | Time    |
|--------------------------------------------|---------|
| today: fresh `from_torch` + `requires_grad=True` | 46.5 us |
| proposed: cached view + `grad.zero_()`      | 16.3 us |

Reusing the wrapper and explicitly zeroing its gradient buffer is **2.9x cheaper** than
rebuilding it, so this is not a forward-only optimisation.

---

# 2. Cross-check against the `warpSPH` frontend

Findings that constrain the design. These are the reason the plan looks the way it does.

## 2.1 The blast radius of an API change

* **24 frontend files** call `warpWrapper2` / `extractStateInfo` directly
  (`modules/adaptiveSupport`, `modules/compSPH`, `modules/crk`, `modules/deltaSPH`,
  `modules/dissipation`, `modules/incompressible`, `modules/liu`, `modules/mdbc`,
  `modules/pressure`, `modules/shockCapturing`, `modules/surfaceDetection`,
  `modules/util`, `sample/wp_deltaShift.py`).
* **65 `warpOperation(...)` call sites** across systems, schemes, and case setup.

`warpSPH` is not public yet, so a breaking change is affordable -- and this is the
right moment for one (Section 8). But the sequencing matters: **Steps A-F land
underneath the existing signature**, so the ~2x win arrives with zero frontend risk,
and the interface change is a separate, later step with its own migration. An API break
is not required for any of the performance work; it is required for `ExecutionMode`
ergonomics and it is the cheapest chance to fix the positional ten-tuple.

## 2.2 The frontend does not use `ParticleState` -- it duck-types it

`warpSPH/systems/baseState.py` defines `BaseParticleState(BaseState)`, a dataclass
with `positions`/`supports`/`masses`/`densities`/`kinds` (plus `materials`, `UIDs`),
and passes **that object itself** as `queryParticles`. There is exactly one
`ParticleState(` construction in the whole frontend `src/`, and it is a re-export.

Conclusion: **`Field` can never be a required member of a state object.** The core
receives plain `torch.Tensor` attributes off objects it does not own and cannot
change. The caching mechanism has to work on bare tensors arriving from outside.

## 2.3 The integrator's update semantics decide the cache hit rate

`warpSPHIntegrators/fields.py` allocates in two places, and the second one dominates:

* `update_component` / `update_position` build the new value out-of-place
  (`value = value * s; value = value + w * ref; value = value + dt * delta`) -- three
  fresh tensors per component per stage.
* `_state_initialize` (`initializeNewState`) calls `clone_value` on **both `constant`
  and `integrated`** fields. So a clone produces fresh tensor objects for `supports`,
  `masses`, `densities`, `kinds`, `materials` and `UIDs` as well as for `positions`
  and `velocities` -- not just the integrated pair.

RK4 takes 5 state clones per step (already the theoretical minimum after v0.5.0's
`copyState=False` fix), so roughly **40 fresh tensor objects per step**, all of them
cache misses on first touch.

Hit profile per step: at the start of each stage the ~8 particle fields miss once
(~56 us total); the ~28 remaining flat entries -- adjacency, domain, and every null
field -- hit. Every subsequent operator call within that stage hits on everything.
So the cost of the integrator's allocation style is on the order of **280 us per RK4
step**, against roughly **9 ms per step saved** by the cache at 20 operator calls.

**The integrator's out-of-place style is therefore not a blocker and not a
prerequisite.** It is worth changing for its own reasons -- see Section 7 -- but
sequencing it before the field work would be optimising blind.

One property worth noting because it makes cloning safe by accident rather than by
care: `torch.Tensor.clone()` does **not** carry python attributes (verified), so a
cloned state arrives with no attached field and correctly rebuilds. The clone path
cannot produce a stale view.

## 2.4 A frontend hazard the design has to defend against

`copy.deepcopy` is used on state objects (`caseUtils/waveEquation/sample.py:42`,
`caseUtils/weaklyCompressible.py:309`) and states reach `pickle` through
`warpSPH/io/hdf5.py`. Verified behaviour of a python attribute attached to a tensor:

| operation            | attribute survives? |
|----------------------|---------------------|
| `t.clone()`          | no                  |
| `copy.deepcopy(t)`   | **yes**             |
| `torch.save`/`load`  | **yes**             |

A `wp.array` is ctypes-backed; deep-copying or pickling one is at best meaningless and
at worst a dangling device pointer. **`Field` must therefore be explicitly
non-copyable and non-picklable** (Section 3.2). This is the single most likely way a
naive implementation silently corrupts a restart file.

## 2.5 `kinds` is already mandatory in practice -- it just is not typed that way

`ParticleState.kinds` is `Optional[torch.Tensor] = None`, and `checkKinds` papers over
the `None` case for `OperationDirection.AllToAll` by handing out an **N-sized** zero
dummy (`getCachedDummyTensor((queryNumParticles,), int32)`), because `getParticle` /
`getParticleData` in `util/stateUtil.py` index `SoA.kinds[i]` unconditionally. For any
other `OperationDirection` it raises. That `None` branch is also the source of the
`kinds=None` + `AllToAll` out-of-bounds read still recorded as open in
`warpier_core.md`.

Surveyed every construction site in both repos plus `scripts/`:

* **Every** core gradcheck script, `scripts/operation_matrix.py`, and `debug_crk_backward.py` passes a real `kinds` tensor.
* **Every** frontend case setup passes one (`caseUtils/**/sample.py`, `initializers/weaklyCompressible.py`, `rigidBody/ghostParticles.py`, `io/importIO.py`, `io/dataset.py`, ...), and `BaseParticleState.kinds` is declared `constant`, so it is always populated.
* The single grep hit for `kinds = None` is `scripts/gradcheck_renorm_native.py:62`, which reads `adjacency, kinds = None, torch.zeros(...)` -- it is *adjacency* that is `None`.

**Not one caller in either repo relies on `kinds=None`.** Making it a required member
is therefore a type-annotation change plus a deletion, with no call-site churn -- and
it removes the N-sized dummy, the `hasattr(queryParticles, 'kinds')` probe in
`arg_extract.py`, the whole `checkKinds` dummy branch, and the out-of-bounds read, all
at once.

This is strictly better than the `hasKinds: wp.bool` struct flag the previous draft
proposed: the struct ABI does not change, so the two frontend kernels that read
`referenceState.kinds` directly (`modules/pressure/wp_surfaceAware.py:178`,
`modules/dissipation/wp_diffusion.py:184`) keep working untouched. The old Step F
disappears.

---

# 3. Design

## 3.1 `Field`: the dual representation

```python
class Field:
    __slots__ = ("views", "tangent", "_ptr", "_shape", "_strides", "_dtype", "_owner")

    views:   dict[Role, wp.array]   # {PRIMAL: <view>}; TANGENT added in Phase 6
    tangent: "Field | None"         # None today -- see Sec. 3.6, requirement 1
    _ptr / _shape / _strides / _dtype   # provenance, for revalidation
    _owner:  torch.Tensor | None    # set ONLY for standalone fields (Sec. 3.4)
```

The `views` map and the `tangent` slot are the two forward-mode affordances from
Section 3.6. Both are free today: one role, one dict lookup, one `is None` check.

Two ownership modes, and the difference matters for lifetime:

**Attached fields** (tensors owned by a caller: everything from the frontend).
The `Field` is stored on the tensor as `t._wsc_field`. It holds **no strong reference
back to `t`** -- only `warp`, which internally holds `t.detach()`, a *distinct* tensor
object that shares storage. Verified: `wa._tensor is t` is `False` and the detached
alias's `_base` is not `t`, so **there is no reference cycle**. Confirmed empirically
with `gc.disable()`: a tensor carrying an attached field dies by refcount alone, while
the same structure holding `t` strongly does not. This is the property that lets us
skip both a global registry and `weakref.finalize` (which measured 22 us per
registration -- three times the cost of the `from_torch` it was meant to save).

**Standalone fields** (core-owned: null fields, core-constructed states, adjacency).
Here `Field` *does* hold the torch tensor in `_owner`, because the whole point is that
the Field is the owner and outlives any particular call.

### Revalidation

```python
def view(self) -> wp.array:
    # ~0.4 us; rebuild only when provenance changed
```

Checked against the live tensor on each acquisition:

* `data_ptr` unchanged -- catches `resize_`, `.data` reassignment, storage swaps.
* `shape` / `strides` / `dtype` unchanged.

Verified semantics that make this sound:

* In-place mutation through the torch side (`copy_`, `add_`) **is visible through the
  cached warp view** and leaves `data_ptr` stable. This is the desired behaviour and it
  is what makes `densities`' `copy_` in `finalize` a cache hit rather than a bug.
* **Non-contiguous tensors are a real staleness hazard and must never be cached.**
  Verified: `castTorchToWarpAsBuiltins` calls `.contiguous()`, which *copies*; a later
  in-place write to the original is invisible through the view, and `data_ptr` does not
  change, so the validity check would report a false hit. The acquisition path must
  therefore refuse to cache when `t.is_contiguous()` is `False` and rebuild every time.
  A test pins this (Section 5, test 5).

### Copy / pickle safety

Per Section 2.4, `Field` defines:

```python
def __copy__(self):            return None
def __deepcopy__(self, memo):  return None
def __reduce__(self):          return (_no_field, ())   # unpickles to None
```

A copied or restored tensor therefore arrives with `_wsc_field = None`, which the
acquisition path treats as a miss and rebuilds. Silent correctness, no cross-process
device pointers.

## 3.2 `acquire`: the one entry point

```python
def acquireView(t: torch.Tensor, role: Role = Role.PRIMAL) -> wp.array
```

Replaces `getCachedWarpArray`. Logic: read `t._wsc_field` (0.060 us) -> revalidate ->
hit returns `field.views[role]`; miss (or non-contiguous, or `None`) builds a fresh
view and, if contiguous, attaches it. `role` is defaulted and unused today; it is
Section 3.6's requirement 2.

Escape hatch: `WARPSPHCORE_DISABLE_FIELD_CACHE=1` forces the miss path unconditionally,
restoring exactly today's semantics. Any bisect of a suspected caching bug is then one
environment variable, not a revert.

## 3.3 Gradient hygiene -- and the bug this must not reintroduce

`autograd/cache.py` carries a long comment explaining why the previous
`data_ptr`-keyed `wp.array` cache was **deleted**: the reused array's `.grad` buffer
was reused too and never zeroed, so any two calls sharing tensor storage accumulated
each other's gradients -- non-reentrant backward, wrong from the second call onward.
`torch.autograd.gradcheck` is precisely the workload that exposes it. **This plan
reintroduces wrapper reuse, so it must discharge that bug explicitly, by design and
not by luck.**

Three things are different this time:

1. **Ownership is explicit, not inferred.** The old cache keyed on `(data_ptr, shape,
   strides, dtype)` in a global dict, so two unrelated tensors that happened to land on
   the same recycled allocator block aliased each other. A field attached to the tensor
   object is valid for exactly one tensor by construction; there is no inference step
   that can be wrong.

2. **Zero-on-acquire, unconditionally.** `StateAwareWarpFunction.forward` zeroes
   `field.warp.grad` for every differentiable input *before* taping, every call. The
   contract is "the gradient buffer is zero at the start of forward", enforced at the
   point of use rather than depending on a `tape.zero()` that ran (or did not run) at
   the end of some earlier backward. Cost: 16.3 us vs 46.5 us for today's
   rebuild-from-scratch, so the safe thing is also the fast thing. The existing
   `tape.zero()` in `backward` stays as belt-and-braces.

3. **It ships behind a gate.** Step C lands view reuse on the **no-grad path only**
   (`WARPSPHCORE_FIELD_CACHE_GRAD=0`), which is where the frontend's simulation
   workload lives and where the 900 us matters. Step D flips it on for the grad path
   only after the full gradcheck suite passes twice-in-process (Section 5).

**Signed off (2026-08-16).** The standing instruction was to rip caching layers out
rather than work around them; the distinction accepted here is between
*inferred-identity* caching (deleted, correctly) and *owned-identity* caching with an
enforced grad-buffer contract. The deciding factor is that the detection machinery that
did not exist when the old cache was written now does:

* `.github/workflows/tests.yml` runs `pytest tests/` -- which includes
  `tests/operations/test_gradcheck_scripts.py`, running each `scripts/gradcheck_*.py`
  in its own subprocess -- plus the operation matrix at five configurations
  (2D float32, 2D float64, 1D, jittered, and 3D when a CUDA runner is available).
* `warpSPH/tests/test_gradcheck_scripts.py` **auto-discovers** `gradcheck_*.py` rather
  than listing them, so a new module cannot silently stop being covered, and
  `tests/test_physics.py` provides larger-scale gradient behaviour checks.

Two gaps in that machinery are worth closing alongside this work, and are folded into
the gates below: the frontend has **no `.github/workflows/` at all** (its tests exist
but nothing runs them automatically), and every gradcheck runs in a *fresh* subprocess,
so nothing currently exercises two calls in one process except gradcheck's own internal
repeated evaluation. The twice-in-process gate in Step E addresses the second directly.

## 3.4 Null fields: retiring the dummy tensors

Today `extractStateInfo` calls `getCachedDummyTensor` up to 8 times per call (1.2 us
each), and each dummy then goes through `from_torch` again (~7 us each) -- so a call
with no corrections enabled spends ~60 us per launch marshalling placeholders that are
never read.

Replace with a permanent registry of **null `Field`s**, built once per `(kind, dim,
device, precision)` and never converted again:

| kind        | shape       | used for                                  |
|-------------|-------------|-------------------------------------------|
| `scalar`    | `(1,)`      | densities, omegas, volumes, CRK `A`       |
| `vector`    | `(1, D)`    | CRK `B`, `gradA`                          |
| `matrix`    | `(1, D, D)` | renormalisation matrices, CRK `gradB`     |
| `int32`     | `(1,)`      | `numCells`, adjacency offsets/counts      |
| `int64`     | `(1,)`      | neighbor list, sort index                 |
| `vec2i`     | `(1, 2)`    | hash table                                |
| `vec3i`     | `(1, 3)`    | cell offsets                              |
| `vec3l`     | `(1, 3)`    | cell table                                |
| `bool`      | `(1,)`      | periodicity                               |

Cost of a disabled correction path drops from ~15 us to a dict lookup.

### "Proper invalid values"

Null fields are zero-filled by default (matching today). Add
`WARPSPHCORE_NULL_FILL=sentinel`, which fills float nulls with NaN and integer nulls
with `INT_MIN`. Every null field is supposed to be unreachable -- `getVolume_i`,
`getGradH_i`, `getCRK_i` and friends in `util/stateUtil.py` all branch on the
`correctionData.useX` flag before indexing. Sentinel fill turns "supposed to be" into a
test: any path that reads a disabled field poisons its output loudly instead of
contributing a plausible zero. Wire it into a CI job that runs the operation matrix and
gradcheck suite with sentinel fill on (Section 5).

## 3.5 `StateBundle`: killing the per-call struct rebuild

`extractStateInfo` currently allocates a fresh closure per call and `build_fn` performs
~51 warp struct field writes (66.5 us). Both are re-derived every launch from inputs
that almost never change.

Introduce a `StateBundle`: a persistent object holding the preallocated
`particleDataSoA_*`, `correctionData_*`, `adjacencyData`, `gridData`, `domainData`, and
`kernelState` instances, cached in a small LRU keyed on the **configuration signature**:

```
(executionMode, dim, useAdjacency, useCRK, useGradHTerms, useVolumes,
 useGradientRenormalization, mode_uint, kernel_int, gradientMode_int,
 laplacianMode_int, positiveDivergence, divergenceMode, opInt, grid_numOffsets)
```

`executionMode` (`NONE | REVERSE | FORWARD`) leads the key, and the struct types are
resolved through `structFor(kind, dim, mode)` rather than the `dim == 1 ? ... : ...`
ternaries currently inlined in `arg_extract.py` -- Section 3.6, requirements 3 and 4.
Adding forward mode then means registering a row in that table, not editing extractors.

`bundle.refresh(views)` compares the incoming `wp.array` objects against what the
structs already hold by identity and writes only what changed -- 1.6 us on a full hit
versus 66.5 us. Scalars that legitimately change every step (`grid_hCell`) are always
assigned; they are single scalar writes.

Small frontend-side simplification worth taking while both repos are in scope: making
`OperationProperties` a **frozen** dataclass makes it hashable, so the bundle can key on
the properties object itself instead of unpacking ten scalars per call. Frontend call
sites construct it inline (`systems/incompressible.py` builds a fresh one inside every
`warpOperation` call) and would not need to change; hoisting those constructions out of
the hot path is then an easy follow-up that turns the key into an identity check.

This subsumes Phase 4 ("Improve State Construction"): the mechanical struct-population
boilerplate becomes one `refresh` call, and the validation currently spread across
`arg_check.py` moves into bundle construction, where it runs once per configuration
instead of once per launch.

## 3.6 Forward-mode readiness: what Phase 6 needs from this design

`warpier_core.md`'s Phase 6 sketches `Field { primal, tangent }`. Two facts, both
verified here, determine what that actually costs and therefore what this design has
to leave room for.

### Fact 1: Warp has no forward-mode AD

Probed `warp 1.15.0` directly: `wp.Tape` exposes `backward`, `get_adjoint`,
`record_launch`, `zero` -- and nothing else. There is no `jvp`, no dual number type, no
tangent propagation anywhere in the package (`spatial_cross_dual` is unrelated
quaternion math). `warpSPHIntegrators/NOTES.md` §2.2 reaches the same conclusion
independently.

So forward mode here is **not** "turn on a warp feature". Every tangent has to be
propagated by code we write. The plan must not imply otherwise.

### Fact 2: every SPH operator is linear in the field values

Measured, not assumed. For `f(a*q1 + b*q2)` vs `a*f(q1) + b*f(q2)` at N=4000, 2D,
float32:

| operator                        | relative max error |
|---------------------------------|--------------------|
| Interpolate (Naive)             | 5.9e-07            |
| Gradient (Difference)           | 3.4e-07            |
| Gradient (Summation)            | 6.4e-07            |
| Laplacian (Brookshaw)           | 2.7e-07            |
| Gradient + renormalisation      | 3.6e-07            |

All at float32 round-off. The correction terms (`L`, CRK `A`/`B`/`gradA`/`gradB`,
`omega`) depend on positions, supports, masses and densities but **not** on the field
values, so linearity survives every correction path.

This splits Phase 6 into two very different tiers:

* **Tier 1 -- tangents w.r.t. field values.** The JVP of a linear map is the map
  itself. Propagating a tangent means **re-launching the same kernel on the tangent
  array**. Zero new kernels, zero new adjoint code, zero new kernel math to validate.
* **Tier 2 -- tangents w.r.t. positions / supports / masses / densities.** The kernel
  is genuinely nonlinear in these (kernel function of `|x_i - x_j| / h`), so this needs
  hand-written JVP twins per operator. This is the expensive tier and it should be
  costed separately, not smuggled in behind Tier 1.

Tier 1 is a plausible first delivery on its own, and this design should make it nearly
free.

### The six structural requirements

Cheap now, expensive to retrofit. Each is a constraint on Steps A and E.

1. **`Field` is pair-capable from day one.** A `tangent: Field | None` slot, `None`
   when forward mode is off. Costs one attribute check per acquisition.
2. **View acquisition is role-keyed.** `acquireView(t, role)` with `role` in
   `{PRIMAL, TANGENT}`; `t._wsc_field` holds a tiny per-role slot rather than a single
   array. One role today, no rewrite later.
3. **Struct types come from a table, never a hard-coded ternary.** `arg_extract.py`
   currently does `particleDataSoA_1 if dim == 1 else ...` inline. Replace with
   `structFor(kind, dim, mode)`. Phase 6 registers `particleDataSoA_2_dual` and
   friends into the same table. **This is the single most important item** -- leaving
   the ternaries in place means Phase 6 rewrites every extractor.
4. **`ExecutionMode` exists now, with `FORWARD` declared and unimplemented.**
   `NONE | REVERSE | FORWARD`, carried on a minimal execution context and folded into
   the `StateBundle` cache key. Phase 6 then adds a value to an enum and a row to a
   table rather than threading a new parameter through every signature. This is what
   `warpier_core.md` means by "forward mode should be a property of the execution
   context rather than individual kernels".
5. **Null fields are the zero-tangent supply.** Forward mode needs a zero tangent for
   every input that is not being seeded -- which is exactly what the Section 3.4
   registry already hands out, at no conversion cost. The "stop making dummy tensors"
   work and the "unseeded tangent" work are the same mechanism. Note the fill policy
   interaction: tangent nulls must be **zeros**, never sentinel NaN, since a zero
   tangent is semantically meaningful rather than a can't-happen placeholder.
6. **Bundle construction stays independent of the AD bridge.** Today
   `extractStateInfo` returns a `build_fn` closure that only `StateAwareWarpFunction`
   can consume. `StateAwareWarpFunction` is a `torch.autograd.Function` and is
   therefore inherently reverse-mode; forward mode needs a *different* bridge that does
   not go through torch autograd's backward machinery at all. Step E already separates
   extraction from the bridge -- this is the reason it must, and the separation has to
   be real rather than cosmetic.

### Recommended forward-mode route (for Phase 6, not this plan)

`torch.autograd.forward_ad` interoperates cleanly: verified that
`fwAD.unpack_dual(d)` returns a primal sharing `d`'s `data_ptr` and a contiguous
tangent. So the boundary is mechanical -- unpack the dual on the torch side, acquire
two views through the same `Field`, launch, re-dual the output with `make_dual`. The
Field's tangent slot maps one-to-one onto torch's dual representation, which means no
bespoke tangent bookkeeping to invent.

## 3.7 Two cheap wins to pick up in passing

* **`record_function` hooks are always on.** 12 enter/exit pairs per operator call,
  94 across the package, costing real time whether or not a profiler is attached.
  Gate them behind a module-level `PROFILING` flag (a no-op context manager when off).
* **`allocateTorchWarp` costs 85 us.** `torch.zeros(20000)` is 26 us of that;
  `wp.dtype_to_torch` (0.15 us) and `wp.device_to_torch` (0.50 us) are small but are
  re-resolved on every call. Memoise the `(warp dtype, warp device) -> (torch dtype,
  torch device, trailing shape, scalar dtype)` mapping. Do not pool the output buffers
  themselves -- autograd requires each output to be a fresh tensor.

---

# 4. Execution plan

Each step is independently landable, independently revertable, and gated. Steps are
ordered so that the riskiest change (D/E) lands after the infrastructure that makes it
testable. Steps 0-B and F are pure wins with no caching risk at all.

## Step 0 -- Baseline harness (prerequisite, no behaviour change)

* Add `scripts/bench_call_overhead.py`: per-stage us/call breakdown (extract, convert,
  build, allocate, launch, total) across `{1,2,3}D x {2k, 20k, 200k} x {adjacency,
  grid} x {no corrections, CRK, renorm}`, plus a grad-path variant and a
  tensors-allocated-per-step counter for Section 7.
* Record the current numbers into `docs/regression/` as the comparison point.
* **Gate:** numbers reproduce Section 1 within noise.

## Step A -- Make `kinds` a required member  [cheap, do it first]

Independent of everything else, and it shrinks every later step.

* `ParticleState.kinds: torch.Tensor` -- no longer `Optional`.
* Delete `checkKinds`'s dummy branch and its N-sized `getCachedDummyTensor` call;
  what remains is a shape/device validation.
* Drop the `hasattr(queryParticles, 'kinds')` probe in `arg_extract.py`.
* `sphOperation_warp`'s `queryKinds`/`referenceKinds` parameters stop defaulting to
  `None` and validate instead.
* No struct ABI change, so the two frontend kernels reading `referenceState.kinds`
  are untouched (Section 2.5).
* **Gate:** operation matrix bit-identical; gradcheck suite green; a new test asserts
  `kinds=None` is now a clear error rather than an out-of-bounds read; frontend suite
  green. Closes the open `kinds=None` + `AllToAll` item in `warpier_core.md`.

## Step B -- `Field`, `acquireView`, and the null-field registry

* New `dataTypes/field_t.py`: `Field`, `Role`, `FieldKind`, `ExecutionMode`, and the
  copy/pickle guards.
* New `util/fieldRegistry.py`: `acquireView`, `nullField`, `structFor`, fill policy,
  and the `WARPSPHCORE_DISABLE_FIELD_CACHE` / `WARPSPHCORE_NULL_FILL` env switches.
* **Do not wire into the hot path yet.** Unit tests only.
* **Gate:** new unit tests (Section 5, tests 4-10) pass; `pytest` green; no perf change.

## Step C -- Null fields replace dummy tensors in `arg_extract.py`

* `getCachedDummyTensor` call sites in `arg_extract.py` and `arg_check.py` become
  `nullField(...)` lookups. With Step A landed, `getCachedDummyTensor` has no
  remaining N-sized caller and can be deleted outright.
* `extractStateInfo` returns `Field`s for the null slots; the 36-entry flat layout and
  its index meanings are unchanged.
* **Gate:** operation matrix (full sweep) bit-identical; gradcheck green; bench shows
  the null-path improvement.

## Step D -- View reuse, no-grad path only  [signed off, Section 3.3]

* `getCachedWarpArray` delegates to `acquireView`.
* `StateAwareWarpFunction.forward` uses cached views **only when
  `ctx.any_requires_grad` is False**; the grad path keeps building fresh wrappers,
  exactly as today.
* **Gate:** operation matrix bit-identical; full gradcheck suite green; reentrancy
  tests (Section 5, tests 1-3) pass; bench shows the conversion loop at ~13 us.

## Step E -- View reuse on the grad path

* Zero-on-acquire for every differentiable input in `forward`, before taping.
* Remove the no-grad gate; keep `WARPSPHCORE_FIELD_CACHE_GRAD=0` as an escape hatch.
* **Gate:** the full gradcheck suite, each script run **twice in the same process** --
  the exact shape of the workload that exposed the original bug. Plus the frontend's 11
  `gradcheck_*.py` scripts, same protocol.

## Step F -- `StateBundle` replaces the per-call closure

* New `autograd/stateBundle.py`. `extractStateInfo` becomes a thin resolver: compute
  the config signature (Section 3.5), fetch or create the bundle, `refresh`.
* `build_fn` disappears; the bridge takes the bundle. `warpWrapper2`'s signature is
  still unchanged at this point -- the interface break is Step H, not this one, so the
  frontend stays untouched through F. Bundle construction must be usable *without* the torch
  autograd bridge -- Section 3.6, requirement 6.
* Replace the inline `dim ==` struct ternaries with `structFor(kind, dim, mode)`.
* Fold in the `record_function` gate and the `allocateTorchWarp` memoisation (3.7).
* Optionally freeze `OperationProperties` so it can key the bundle directly.
* **Gate:** operation matrix bit-identical; gradcheck green; bench shows total fixed
  overhead at or below 500 us/call; a debug assertion confirms a refreshed bundle
  equals a freshly built one.

## Step G -- Forward-mode readiness audit (no forward mode implemented)

Not a feature step -- a checklist run against the landed code, so Phase 6 starts from a
known state rather than a hopeful one.

* `ExecutionMode.FORWARD` exists and is rejected with a clear error at every entry.
* No `dim ==` struct ternary survives outside `structFor`.
* Bundle construction has no import-level or call-level dependency on
  `torch.autograd.Function`.
* A throwaway spike proves Tier 1 (Section 3.6): seed a tangent on `queryValues`,
  launch the *existing* kernel on the tangent arrays, and check the result against
  `torch.autograd.functional.jvp` on a small case. If that spike passes, Tier-1
  forward mode is a bridge, not a kernel project.
* **Gate:** all four hold; findings written back into `warpier_core.md`'s Phase 6.
  Requirement 4's *API-surface* half is Step H's job -- G audits the internals,
  H exposes them.

## Steps H / I -- The interface break

Detailed in Section 8, and deliberately last: land, measure and stabilise the
internals before the interface they sit behind starts moving. Step H introduces
`OperatorSpec` / `SPHContext` / `launchOperator` and ports the core, with
`warpWrapper2` reduced to a shim so the frontend is untouched; Step I migrates the
frontend's 24 wrapper sites and 65 `warpOperation` sites and deletes the shim.


# 5. Verification

## Correctness

1. **Reentrancy (the regression that matters).** Call each operator twice on the same
   leaf tensors in one process; compare both gradients against a fresh-process
   single-call run. This is what the deleted cache failed.
2. **Full gradcheck suite** via the `gradcheck` skill: Density, Interpolate, Gradient,
   Divergence, Curl, Laplacian, plus `gradcheck_crk_native`, `gradcheck_renorm_native`,
   `gradcheck_covariance_native`, `gradcheck_pinv_native`,
   `gradcheck_scalar_arg_native`. Each **run twice in the same process** from Step D on.
3. **Frontend gradchecks:** the 11 `warpSPH/scripts/gradcheck_*.py`, same protocol.
4. **In-place visibility:** `copy_` / `add_` through the torch side is observed by the
   cached view; `data_ptr` stability asserted.
5. **Non-contiguous refusal:** a strided tensor is never cached; a write to it is
   observed on the next acquisition. (Pins the hazard verified in Section 3.1.)
6. **Lifetime / no leak:** with `gc.disable()`, a tensor carrying an attached field is
   collected by refcount alone; the storage is released with it.
7. **Copy/pickle safety:** `copy.deepcopy(tensor)` and a `torch.save`/`load` round trip
   yield a tensor whose next acquisition rebuilds rather than reusing a stale pointer.
8. **`kinds` is mandatory:** constructing a `ParticleState` without `kinds` raises;
   `AllToAll` at `N > 1` no longer reads out of bounds (the previously open item).
9. **Tangent-slot inertness:** with `Field.tangent is None` and one role registered,
   acquisition results are bit-identical to a Field type without those slots -- the
   forward-mode affordances cost nothing and change nothing.
10. **Null-field isolation:** with `WARPSPHCORE_NULL_FILL=sentinel`, the whole operation
   matrix and gradcheck suite still produce finite results -- proving no code path
   reads a disabled correction.
11. **Forward equivalence:** `scripts/operation_matrix.py` full sweep (precision x dim x
   jitter x device x traversal x correction) before/after, bit-identical.
12. **Frontend physics:** `warpSPH/tests/test_physics.py`, plus `sod_2d` and a
    Taylor-Green run compared against stored reference output.

Note the platform constraint on the sweeps: 3D workloads run on GPU only -- warp's CPU
backend is single-core and prohibitively slow for them.

## Performance

`scripts/bench_call_overhead.py` reports the per-stage breakdown at every gate. Exit
criteria for the whole effort:

* conversion loop: 288 us -> < 20 us (steady state, all hits)
* struct assembly: 66.5 us -> < 5 us
* total fixed per-call overhead: ~900 us -> < 500 us
* grad-path per-tensor cost: 46.5 us -> < 20 us
* no regression in GPU kernel time at N=200k (this plan must not touch kernel math)

---

# 6. Risks, and the decisions behind this plan

| Risk | Severity | Mitigation |
|------|----------|------------|
| Reintroducing the gradient-accumulation bug that got the last cache deleted | **High** | Owned rather than inferred identity; zero-on-acquire contract; no-grad-only in Step C; twice-in-process gradcheck as the Step D gate; `WARPSPHCORE_FIELD_CACHE_GRAD=0` escape hatch. **Needs sign-off before Step C.** |
| `deepcopy`/`pickle` of a tensor carrying a device pointer corrupts a restart file | **High** | `Field.__deepcopy__`/`__copy__`/`__reduce__` all degrade to `None`; test 7. Hazard is live today in `caseUtils/waveEquation/sample.py` and `io/hdf5.py`. |
| Non-contiguous tensor cached, later in-place write invisible | Medium | Never cache when `is_contiguous()` is false; test 5. Verified as a genuine staleness path, not a theoretical one. |
| Attaching an attribute to caller-owned tensors is a side effect on someone else's object | Low | Invisible to torch semantics, does not survive `clone()`, dies with the tensor, and the repo already does this in `launcher.py`. Disableable via env var. |
| `StateBundle` LRU key misses a field that actually varies | Medium | Key derived mechanically from the same `cfg` dict `arg_extract` already builds; assert in debug builds that a refreshed bundle equals a freshly built one. |
| Making `kinds` mandatory breaks an unsurveyed caller | Low | Surveyed both repos plus `scripts/`: zero callers pass `kinds=None` (Section 2.5). Fails loudly at construction, not silently. |
| Forward-mode affordances turn out to be the wrong shape for Phase 6 | Low | They are three slots and a lookup table, not an implementation. Step G's Tier-1 spike tests the assumption before anything is built on it. |
| Tier-2 forward mode (tangents w.r.t. positions) is mistaken for cheap because Tier 1 is | Medium | Section 3.6 separates them explicitly and costs Tier 2 as hand-written JVP twins per operator. Do not let the linearity result imply otherwise. |

## Decisions -- resolved 2026-08-16

1. **Reintroducing wrapper reuse (Section 3.3): approved.** CI and the frontend's
   larger-scale gradient tests now catch reuse, non-reentrancy and silent gradient
   bugs that were invisible when the previous cache was written. Recorded in 3.3.
2. **`WARPSPHCORE_NULL_FILL`: zeros by default, sentinel available** and exercised in
   CI. Note the Section 3.6 caveat -- tangent nulls stay zeros in every mode.
3. **Integrator buffer pool: after this plan** (Section 7). Interface requirement
   stands now: pooled buffers must carry permanent Fields, and the in-place switch must
   be the same `ExecutionMode` switch, not a second one.
4. **Forward mode: after this plan.** Phase 6 is not a deliverable here; Step G's
   readiness audit and Tier-1 spike are, so Phase 6 starts from a tested assumption
   rather than a hopeful one.
5. **API break: approved, and sequenced last** (Section 8). Land the internals first,
   then the interface, then migrate the frontend.

---

# 7. The integrator's allocation style: a follow-on, not a prerequisite

`warpSPHIntegrators/NOTES.md` §2.1 already diagnoses this and already recommends a
direction. This section only adds what the field work changes about it, and answers
the sequencing question.

## 7.1 What is actually happening

Per Section 2.3: the update helpers build values out-of-place in three steps, and
`_state_initialize` clones `constant` fields as well as `integrated` ones. RK4 lands
at ~40 fresh tensor allocations per step. NOTES.md §2.1 puts it plainly -- the design
is "functional and allocating ... the opposite of what Warp wants (preallocated
buffers, in-place kernel writes, CUDA-graph capture)", and at 1e6-1e7 particles the
allocation traffic dominates.

The out-of-place style is deliberate and correct as a default: stages must not
overwrite state that later stages still read, and out-of-place arithmetic is what
makes the whole thing differentiable through time.

## 7.2 Why it should come after, not before

Measured (Section 2.3): the integrator's allocation style costs the field cache roughly
**280 us per RK4 step** in first-touch misses, against roughly **9 ms per step** that
the cache saves. It is a ~3% drag on the win, not a blocker.

Doing the buffer pool first would mean building it with no instrumentation for what it
buys and no `Field` to hang the pooled buffers off. Doing it second means:

* Step 0's harness already counts allocations per step, so the benefit is measurable
  before the work starts.
* A pooled buffer has a **stable tensor object for the life of the run**, so it carries
  a permanent attached `Field` -- pool and cache become one mechanism, and the hit rate
  goes to 100%.
* The `axpy` kernel NOTES.md §2.1 wants is itself a warp kernel taking arrays that the
  pool's Fields already hold views of, so it needs no marshalling at all.

**Recommendation: after Steps A-F.** The one thing to fix *now* is the interface: the
pool design should assume buffers carry Fields, so Section 7.3 is additive rather than
a second refactor.

## 7.3 Sketch, aligned with NOTES.md §2.1

* A `PooledState` / `WarpState` base whose `initializeNewState` draws from a per-step
  buffer pool instead of calling `clone_value`. Stage count is fixed per scheme, so the
  pool is sized once at setup.
* Each pooled buffer is allocated once and carries a permanent attached `Field`; the
  warp view is built once for the entire run.
* A fused `axpy` warp kernel (`x = s*x + w*ref + sum_i dt_i * k_i`, variadic in `k`)
  collapses the three-tensor accumulation into one launch.
* Fixed step shape makes the step CUDA-graph-capturable, which is where the remaining
  per-launch overhead goes to die.

## 7.4 The constraint that ties this to forward mode

In-place accumulation into pooled buffers breaks reverse-mode autograd, and it breaks
**forward mode for the same reason**: if forward mode rides on `torch.autograd.forward_ad`
dual tensors (Section 3.6's recommended route), then out-of-place arithmetic propagates
tangents *for free* -- `value + dt * delta` on a dual tensor carries the tangent
automatically -- whereas an in-place write into a raw pooled buffer silently drops it.

So the pool cannot be a global switch. It has to be gated on execution mode:

| mode | integrator path |
|------|-----------------|
| `NONE` (plain simulation) | pooled, in-place, graph-capturable -- the fast path |
| `REVERSE` | out-of-place, as today |
| `FORWARD` | out-of-place, as today; tangents ride the dual tensors |

That is one more reason `ExecutionMode` should exist before the pool work starts, and
it is why NOTES.md §2.1's "out-of-place-vs-in-place switch" and this plan's
`ExecutionMode` should be **the same switch**, not two.

## 7.5 Also worth folding in

NOTES.md §2.2 leaves open "torch autograd vs `wp.Tape` as the authoritative gradient
model for a Warp state". This plan answers it for the operator layer -- torch autograd
is authoritative, `wp.Tape` is an implementation detail inside
`StateAwareWarpFunction` -- and Step F's separation of bundle construction from the AD
bridge is what keeps that answer from being load-bearing. Worth recording the decision
in NOTES.md when the integrator work starts, rather than re-deriving it.

---

# 8. The interface break: a declared operator ABI

Approved because `warpSPH` is not public yet, and because the codebase has done this
once already: the state-based rewrite turned `sphOperation_warp` into a thin adapter
over `warpOperation` and migrated callers behind it (`operations.py`'s own header
comment records the reasoning). The same shape applies here.

## 8.1 What is wrong with `warpWrapper2` today

```python
warpWrapper2(
    launcher, kernel, outputSizes, outputDtypes,
    defaultStateArguments: tuple,     # positional 10-tuple, None-padded
    additionalArguments: tuple = (),  # untyped; tensors/scalars split by isinstance
    numThreads = None,
)
```

* **The ten-tuple is positional and None-padded.** Every one of the 24 frontend call
  sites spells out `(queryParticles, operationProperties, domain, queryVolumes,
  referenceVolumes, adjacency, referenceParticles, crkState, gradHState,
  renormalizationState)`, mostly `None`. A transposed pair is a silent wrong answer.
* **`additionalArguments` is re-analysed every call.** `warpWrapper2` rebuilds
  `add_tensor_pos` / `add_scalar_map` and a fresh reconstruction closure per launch,
  from a declaration that never changes.
* **Output dtype is computed by converting a tensor.** 27 call sites across the two
  repos do `outputDtype = castTorchToWarpAsBuiltins(queryParticles.densities).dtype` --
  a full `wp.from_torch` on the hot path purely to read a dtype off the result. Some
  multi-output sites do it twice.
* **There is nowhere to put `ExecutionMode`**, so forward mode has no entry point that
  is not a global or a thread-local.
* **The real kernel ABI is enforced by comments.** Every operator kernel opens with
  `queryState, referenceState, domainState, useAdjacency, adjacencyState, gridState,
  correctionData, kernelProperties` under a `# Do not change the parameters above`
  banner and closes with `# The last parameter is always the output array`. That is a
  contract; it should be declared, not commented.

## 8.2 The proposed shape

Static, declared once at import time next to the kernel:

```python
@dataclass(frozen=True)
class OperatorSpec:
    kernel:     wp.Kernel
    outputs:    tuple[OutputSpec, ...]    # (dtype, shape source), resolved once
    extras:     tuple[ExtraSpec, ...] = ()  # names + kinds (TENSOR | SCALAR | ENUM)
    threads:    ThreadSpec = ThreadSpec.QUERY_COUNT
```

Per call, cheap to build and natural to hold onto:

```python
@dataclass
class Corrections:                       # collapses 5 of the 10 positional slots
    volumes: tuple[Tensor | None, Tensor | None] = (None, None)
    crk:     CRKState | tuple[CRKState, CRKState] | None = None
    gradH:   GradHState | None = None
    renorm:  RenormalizationState | None = None

@dataclass
class SPHContext:
    query:       ParticleState
    properties:  OperationProperties
    domain:      DomainDescription
    adjacency:   AdjacencyListWarp | CompactHashMap | None = None
    reference:   ParticleState | None = None
    corrections: Corrections = EMPTY_CORRECTIONS
    mode:        ExecutionMode = ExecutionMode.AUTO

def launchOperator(spec: OperatorSpec, ctx: SPHContext, **extras) -> Tensor | tuple[Tensor, ...]
```

A call site collapses from roughly twenty-five lines to three:

```python
_OMEGA = OperatorSpec(kernel=computeOmega_Kernel,
                      outputs=(OutputSpec(scalar_t, ShapeOf.QUERY),))

def computeOmegaWarp(ctx: SPHContext):
    return launchOperator(_OMEGA, ctx)
```

`ExecutionMode.AUTO` resolves to `REVERSE` when any input requires grad and `NONE`
otherwise -- exactly today's behaviour, so no caller has to think about modes until
forward mode exists.

## 8.3 What the new shape buys

* **Named, non-positional state.** No None-padding, no transposition hazard.
* **Extras declared once**, so the per-call isinstance split and index bookkeeping in
  `warpWrapper2` disappear entirely.
* **Output dtypes resolved at import**, retiring all 27 hot-path dtype probes.
* **A home for `ExecutionMode`**, satisfying Section 3.6's requirement 4 at the API
  surface rather than only internally.
* **A caller-held handle.** An `SPHContext` that outlives a call is a natural identity
  key for the `StateBundle`, turning signature computation into a pointer comparison --
  and it is the object a future graph-capture path would be captured against.
* **The kernel ABI becomes declarative**, so the `# Do not change the parameters above`
  banner is enforced by `OperatorSpec` rather than by discipline.

## 8.4 Migration

Same idiom as the state-based rewrite, which is why it is known to work here.

**Step H (core).** Introduce `OperatorSpec` / `SPHContext` / `launchOperator`. Port all
15 core files that use `warpWrapper2` / `extractStateInfo`. Reimplement `warpWrapper2`
as a thin deprecating shim over `launchOperator` -- it keeps its exact current
signature, so **the frontend keeps working untouched throughout this step**. Gate:
operation matrix bit-identical; gradcheck green through both the new path and the shim;
bench shows no regression and the dtype-probe saving.

**Step I (frontend).** Migrate the 24 `warpWrapper2` sites and the 65 `warpOperation`
sites, module by module, each with the frontend gradcheck and physics suites green.
Convert the sprawling `Optional[...]` parameter lists on wrappers like
`computeCompSPHBalanceTermWarp` into an `SPHContext` plus declared extras while doing
so. Then delete the shim and the old `defaultStateArguments` path from the core.

**Prerequisite for Step I:** give `warpSPH` a `.github/workflows/` that runs its
existing `tests/` -- the suites are there but nothing runs them automatically, and a
65-site migration should not be the first thing to find that out.

Both steps are behind everything else deliberately: the perf work should be landed,
measured and stable before the interface it sits behind starts moving.
