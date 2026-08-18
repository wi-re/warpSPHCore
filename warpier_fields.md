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

**Result (Steps A-F landed, 2026-08-17): exceeded on the no-grad path, on target on the grad
path.** Measured on this machine, dim=2, `WARPSPHCORE_PROFILING=0`: no-grad total 1197-1413 us ->
219-316 us (`bench_call_overhead_step_f.md`), well past the ~450-500 us target -- the no-grad path
is where Steps C, D and F's caching all stack (null fields, view reuse, and struct-bundle reuse are
all unconditionally active there). Grad-path total 1682-1995 us -> 346-415 us
(`bench_call_overhead_step_e.md`; Step F does not further reduce it, by design -- see that step's
notes), landing inside the target band. The asymmetry is intentional, not a shortfall: every
caching layer in Steps D-F is gated to `not ctx.any_requires_grad` for correctness reasons specific
to that layer (an unzeroed `.grad` buffer for Step D/E without Step E's fix; a stale struct-field
pointer for Step F), so the grad path pays full reconstruction cost on the struct-assembly side
while still getting Step E's tensor-view-reuse win.

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
* `warpSPH/tests/test_gradcheck_scripts.py` runs 15 `gradcheck_*.py` scripts, one per
  kernel-bearing module, same subprocess-per-script shape as core's version, and
  `tests/test_physics.py` provides larger-scale gradient behaviour checks. **Correction
  (2026-08-17): this list is hardcoded (`GRADCHECK_SCRIPTS = [...]`), not auto-discovered** --
  verified directly against the frontend repo when running its suite as part of Steps A-F's
  landing. The original claim here (and its "only the frontend's does" echo further down) was
  never actually checked against that file; both are wrong. Neither core's nor the frontend's
  gradcheck suite currently auto-discovers -- a new script in either `scripts/` directory needs a
  manual addition to the corresponding `GRADCHECK_SCRIPTS` list, same as core's
  `gradcheck_twice_in_process.py` needed (Step E's notes, above).

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
  **Confirmed end-to-end by Step G's spike** (`scripts/spike_forward_mode_tier1.py`),
  not just inferred from the linearity table above: 14/14 operator x scheme x dim x
  field-rank x correction cases satisfy the JVP identity to float64 round-off, with
  `f(0) == 0` exactly (linear, not merely affine -- an affine term would offset every
  tangent). Stronger than requirement 3 assumed: Tier 1 needs **no new struct type**,
  hence no `FORWARD` row in `structFor` and no widening of the bundle cache key.
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
   **Two corrections from Step G's audit.** (a) Registering a dual struct in the table is
   *not* the whole change if dual structs ever happen: `util/stateUtil.py` carries three
   concretely-typed `@wp.func` overload sets (`getParticleData`,
   `getParticleCorrectionData_i/_j`), one per dim, resolved by **warp's** overload dispatch
   on struct type rather than by any Python branch, so each new struct type needs its own
   set. (b) That cost is avoidable: Tier 1 needs no new struct type at all, so this item is
   insurance for Tier 2 / a fused dual-struct design rather than a Tier-1 prerequisite --
   which downgrades it from "the single most important item" to "cheap and still worth
   having". The table did earn its keep anyway: it is where `FORWARD` gets rejected.
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

**Two things Step G established about that route, one of them a trap.** Passing a dual
tensor through the *current* bridge raises `NotImplementedError: You must implement the jvp
function for custom autograd.Function to use it with forward mode AD` -- clean, and it names
the deliverable precisely. But **`torch.autograd.functional.jvp` must not be used as the
reference** when validating that bridge: it does not raise, it returns a **silently zero**
tangent (its double-backward trick needs a differentiable backward, which reading gradients
off a `wp.Tape` is not, and the default `strict=False` degrades to zeros; `strict=True`
raises). A half-finished bridge that also emitted zeros would agree with it perfectly. Use a
reverse-mode Jacobian contracted with the tangent instead -- exact on small cases, and it
leans only on the first-order backward the gradcheck suite already validates. Step G's text
originally specified the `jvp` reference; `scripts/spike_forward_mode_tier1.py` documents
and probes both failure modes so the next person does not have to rediscover them.

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

## Status as of 2026-08-18

**✅ COMPLETE:** Steps 0 (bench harness), A (kinds required), B (Field/nullField/structFor), C (null
fields wired into `arg_extract.py`), D (view reuse, no-grad path only), E (view reuse, grad path),
F (`StateBundle`, no-grad path only), G (forward-mode readiness audit -- found and fixed two real
bugs, see its section), H (real-workload bottleneck audit -- the ~7% end-to-end observation is now
fully explained; see its results subsection), I (declared operator ABI -- `OperatorSpec` /
`SPHContext` / `launchOperator`, all 9 core call sites ported, `warpWrapper2` reduced to a shim;
see its own subsection under "Steps I / J" above), J (frontend migration -- all 24 `warpWrapper2`
call sites in `warpSPH` ported to `launchOperator`; the 65 `warpOperation` sites needed no change,
already routed through Step I's ported backends; `warpWrapper2` kept, not deleted, per a scope
correction found during this step -- see its own subsection)
- Baseline numbers recorded and gate passes
- Field attachment proven safe; no reference cycles, survives clone, dies with tensor
- **Design decision used:** Option A (flat_tensors carries `[torch.Tensor | Field]`;
  `StateAwareWarpFunction.forward` branches on `isinstance(item, torch.Tensor)` -- the complement
  of the isinstance-Field check the draft specified, same effect)
- **Step C corrections found during implementation** (the draft's audit was written against a
  slightly stale picture of the code):
  1. **`arg_check.py`'s `checkInputRenormalization` / `checkInputGradHTerms` / `checkInputVolume`
     / `checkInputCRK` / `checkQV` are dead code** -- grepped every call site in both this repo;
     none exist. `arg_extract.py` does its own inline None-checks and never calls them. They were
     **not** touched: converting unreachable functions serves no perf goal, and the plan's
     description of them as "called from the correction-state checkers" does not hold for the
     code as it stands. `checkKinds` (Step A, genuinely called) is unaffected.
  2. **`getCachedDummyTensor` was *not* deleted.** `coreOperations/wp_interpolate.py:182-183` is a
     live call site outside `arg_extract.py`/`arg_check.py` (builds a dummy `gradA`/`gradB` for a
     CRKState that only supplies `A`/`B`). It doesn't need a Step-C-style fix: the tensor object it
     returns is stable across calls (cached by shape/dtype/device key inside `getCachedDummyTensor`
     itself), so once Step D's view-reuse cache lands it will get the same win automatically, no
     separate change required. `getCachedIdentityMatrices` and the dummy-tensor cache in
     `cache.py` stay.
  3. **The plan's Step C code sketch for `StateAwareWarpFunction.forward` was incomplete.** It
     showed only the conversion loop, but `flat_tensors` becoming heterogeneous also breaks
     `ctx.save_for_backward(*flat_tensors)` (only accepts real tensors) and `backward()`'s
     `zip(ctx.warp_arrays, ctx.saved_tensors)` (lengths would no longer match once Field entries
     are excluded from `saved_tensors`). Fixed by tracking `ctx.is_tensor` (a per-position bool
     tuple) alongside `flat_tensors`, saving only the Tensor-typed entries, and re-expanding
     `backward()`'s returned gradient tuple against `is_tensor` so Field positions always
     contribute `None`. See `stateAwareWarpFunction.py` for the result.
- **Verification run (2026-08-17, CPU only -- see compute-sharing note below):** full CPU-only
  pytest suite (55 tests, `tests/` minus `test_gradcheck_scripts.py`'s subprocess runs), all 12
  `scripts/gradcheck_*.py` scripts individually (all hardcode `DEVICE = torch.device("cpu")` via
  `_gradcheck_common.py`), and both suites repeated under `WARPSPHCORE_NULL_FILL=sentinel` (Test
  10) -- all green, including CRK/renorm/gradH paths that exercise every `FieldKind` this step
  touches. New regression test: `tests/operations/test_null_field_wiring.py` (disabled-correction
  slots are `Field` instances; the same instance comes back on a second call; the unused
  traversal side -- adjacency vs. grid -- is null; real tensors still flow through unwrapped).
- **CUDA bench recorded (2026-08-17):** `docs/regression/bench_call_overhead_step_c.md` vs. the
  Step-0 baseline, same script/device/grid. dim=2, no-grad, reference configuration: convert
  298.7-319.8 us -> 62.3-64.0 us (~4.7-5.0x), total 1197-1413 us -> 363-454 us (~3.1-3.3x). This
  is Step C's contribution alone (disabled-correction slots stop re-converting); Step D's
  view-reuse for the *real* tensors is what closes the rest of the gap to the ~450-500 us
  end-state target. One sweep-position outlier noted and explained in that doc (not reproduced in
  isolation; looks like a one-time kernel-cache compile, not a regression).

**Step D notes:**
- **Design decision used:** the plan's sketch (`getCachedWarpArray(t, use_cache)` gated by
  `not ctx.any_requires_grad`) was implemented as written, with one correctness fix found before
  it shipped: the existing call site passed `t.detach()` into the conversion helper, and
  `Tensor.detach()` returns a **new object every call** -- attaching the Field-registry cache entry
  to that transient object would mean it was never seen again, so the cache would silently never
  hit (0% hit rate, correctness unaffected, entire point of the step lost). Fixed by passing the
  caller's original tensor `t` through and detaching only at the point of actual conversion, inside
  `getCachedWarpArray`/`acquireView`. Caught by writing a reuse test before trusting the bench
  number, not by the bench number itself (a 0%-hit-rate cache is invisible to a total-time
  measurement if nothing else regresses).
- **WARPSPHCORE_FIELD_CACHE_GRAD was not added.** The plan called for it at this step "so grad-path
  bisects can force fresh builds," but at Step D the grad path is unconditionally uncached
  regardless of any env var (the gate is `not ctx.any_requires_grad`, not a flag) -- an inert
  environment variable would be dead configuration. It's introduced in Step E, where it starts
  actually controlling behavior.
- **Verification:** full pytest suite including CUDA-parametrized tests (90 passed), all 13
  gradcheck scripts both directly and via `test_gradcheck_scripts.py`'s hardcoded
  `GRADCHECK_SCRIPTS` list (this repo's version lists rather than auto-discovers -- only the
  frontend's does, per Section 3.3's CI-wiring note; corrected here after writing it wrong the
  first time), a 2D float32 operation-matrix smoke sweep (258 OK / 0 HIGH-ERR-NAN), and a new
  `tests/operations/test_view_reuse.py` covering Section 5's reentrancy tests 1-3: repeated
  no-grad calls hit the same cached view, the `WARPSPHCORE_DISABLE_FIELD_CACHE` escape hatch still
  works, and -- the interaction Step D specifically introduces -- toggling `requires_grad` on the
  *same* tensor object (no-grad call, then grad call, then no-grad again) produces the right
  forward values and a gradient cross-checked bit-for-bit against a fully-cache-disabled run.
- **CUDA bench recorded:** `docs/regression/bench_call_overhead_step_d.md`. dim=2, no-grad:
  convert 62-64 us (Step C) -> 24-25 us (Step D), ~12-13x over Step C alone and ~12-13x over
  baseline's 291.8-319.8 us; total 1197-1413 us (baseline) -> 275-374 us (Step D), 3.8-4.4x. Grad
  path numbers are within noise of Step C's, as expected -- Step D deliberately does not touch it.

**Step E notes:** see the dedicated subsection under "## Step E" below for the full account --
summary: two real gradient-correctness bugs (grad-buffer double-counting when one tensor fills
two kernel roles; a latent missing `.detach()` in `acquireView`'s fallback branches) were found
and fixed before this shipped as default-on, both caught only by a new
`scripts/gradcheck_twice_in_process.py` gate (now permanently wired into
`test_gradcheck_scripts.py`) that no existing test provided. Full verification battery green
after the fixes; CUDA bench in `docs/regression/bench_call_overhead_step_e.md`.

**Step F notes:** see the dedicated subsection under "## Step F" below -- summary: the plan's
`StateBundle` sketch proposed unconditional sharing of a persistent, mutable struct instance
across calls, with no gating. Verified directly against warp 1.16.0 *before* writing any
implementation code (given the pattern from Steps C-E) that this would have been a correctness
bug far more serious than any of those: `wp.Tape` does not snapshot a struct's field values at
launch time, it holds a live reference and re-reads them lazily at `backward()` time, so sharing a
mutable bundle across grad-requiring calls would silently corrupt an earlier call's gradient any
time its backward is deferred past a later call's refresh of the same bundle -- ordinary PyTorch
usage (build a graph across several ops, call `.backward()` once), not an edge case. `StateBundle`
reuse is therefore gated to `not ctx.any_requires_grad`, exactly like Step D's view-reuse cache but
for a stronger reason: there is no zero-on-acquire-style contract that would make grad-path struct
sharing safe, so unlike Steps D/E there is no escape-hatch env var for it either -- the grad path
is simply always fresh, matching the original `build_fn` untouched. Two further bugs surfaced while
folding in Section 3.7's "cheap wins" (a circular import; an unhashable `wp.Device` used as a dict
key) -- see the Step F subsection for both. Full verification battery green; CUDA bench in
`docs/regression/bench_call_overhead_step_f.md`.

**Step numbering note (2026-08-17):** a new Step H (real-world bottleneck audit) was inserted
between the forward-mode readiness audit and the interface break, at the user's request, after a
real 70k-particle simulation showed a much smaller end-to-end speedup than the per-call
microbenchmarks predicted (see that step's own rationale). The old Steps H and I (the interface
break, Section 8) are renumbered to I and J throughout this document; nothing about their content
changed, only their labels.

**Frontend verification (2026-08-17, `~/dev/warpSPH`, same `warp` conda env, editable install --
so it picked up every Steps A-F change automatically with no reinstall):** ran the frontend's full
test suite against the modified core. **88 passed, 1 skipped, 0 failed:**
- `tests/test_caseSpec.py` + `tests/test_runner.py`: 22 passed, 1 skipped.
- `tests/test_gradcheck_scripts.py`: 15 passed (float64 subprocess-per-script, one per
  kernel-bearing frontend module -- `compSPH`, `dissipation`, `crk`, `adaptiveSupport`,
  `deltaSPH`, `shockCapturing`, `mdbc`, `incompressible`, `liu`, `surfaceDetection`, `util`,
  `deltaShift`, `sdf`, `scalarArg_dt`, `wp_surfaceAware`). This is much broader real-world
  `warpWrapper2`/`warpOperation` call-site coverage than core's own test suite reaches on its own,
  and it's exactly the surface Steps A-F's caching sits underneath.
- `tests/test_physics.py`: 51 passed (sod shock tube 1D/2D/3D, Taylor-Green vortex, dambreak,
  Sedov blast, every compressible solver scheme, uniform-lattice density checks -- real
  simulations, not just single-kernel checks).

Empirically confirms Section 2.1's "Steps A-F land underneath the existing signature, so the ~2x
win arrives with zero frontend risk" claim, rather than leaving it as an assumption. Also caught
the frontend's own `tests/test_gradcheck_scripts.py` docstring/list is the same hardcoded-list
shape as core's, correcting this doc's earlier claim of frontend auto-discovery -- see the
Section 3.3 correction above.

---

## Resuming from here

**State as of 2026-08-17 (re-verified same day, working tree clean at `004e22f`, "finish write
up"): Steps 0, A-H complete, verified, and fully committed.** `git log` for the six commits:
`1418d5f`/`0ecd9d5` (plan + first field steps), `ba3dbee` (Step C-F write-up), `d6c0d2d` ("First
steps of the warpier fields setup" -- Steps 0/A-F code), `caa1232`/`6d3863c`/`004e22f` (Step G's
spike, Step H's real-workload audit and the two host-sync fixes, and this document's write-up).
Nothing is sitting in the working tree uncommitted. The two earlier versions of this paragraph
said first "none of it is committed yet" and then "Step G's changes are in the working tree" --
both were true when written and both are now stale; this is the third correction in a row on the
same claim, which is itself the reason to check `git log`/`git status` directly rather than
trust this paragraph on faith next time either.

**Independently re-run and confirmed on this machine (2026-08-17):** full `pytest tests/` --
**118 passed, 1 skipped** (119 collected; the one skip is
`test_renormalization_does_not_read_back[cpu]`, an intentional internal `pytest.skip` since
that test counts CUDA readbacks and is meaningless on a CPU tensor -- not a gap). That total is
Step G's reported 112 plus Step H's 7 new cases in `tests/operations/test_no_host_sync.py`, so it
reconciles with every "Verification" note upstream in this document; no step's number needs
correcting. All five CI-gated `operation_matrix.py` configurations (2D float32, float64, 1D,
light jitter, 3D-CUDA) reproduced **258 OK / 0 HIGH-ERR-NAN** exactly as claimed throughout Steps
D-H. All 14 gradcheck scripts plus the Tier-1 spike passed via `test_gradcheck_scripts.py` (part
of the same `pytest tests/` run). GPU headroom checked before running (4.3/24.6 GB used, ~6%
utilization on this run) per the GPU-sharing note below. If you're a fresh session: read this
whole "Status as of 2026-08-17" block plus each Step's own subsection (B through H) before
touching anything -- they carry the corrections and hazards below, and the numbered lists under
"Step X notes" are denser than the surrounding prose for a reason.

**Step I is done as of 2026-08-18 (this machine, working tree should be clean at whatever commit
follows "Step I: port Interpolate/Gradient/Divergence/Curl/Laplacian to launchOperator" -- check
`git log` rather than trusting this sentence, per the habit established above).** Its own
subsection is under "Steps I / J" earlier in this section -- read it before touching
`autograd/operator_spec.py`, any `coreOperations/*.py` backend, or any `crk/*.py` backend; it
carries the `callable()` vs `types.FunctionType` bug, the GPU-mismatch benchmarking hazard, and why
each vector-output operator's dtype resolver reads `flatOutputShape` back out of `extras` instead
of recomputing it.

**Step J is done as of 2026-08-18** (this machine; `~/dev/warpSPH` working tree has all 24 module
files modified, plus one core change to `autograd/operator_spec.py` in this repo -- check `git
status`/`git log` in both repos rather than trusting this sentence). Read its own subsection under
"Steps I / J" above before touching `operator_spec.py`'s `numThreads` field, any of the 24 ported
frontend modules, or the `warpWrapper2` shim -- it carries three corrections against the plan's
original Step J text: the 65 `warpOperation` sites needed no work (already routed through Step I),
a real `ThreadSpec`/`numThreads` gap in `OperatorSpec` that `compSPH/balance.py` exposed, and why
`warpWrapper2` was deliberately *not* deleted (three live non-module callers in `scripts/` across
both repos). The CI prerequisite (a `.github/workflows/` in `warpSPH`) was explicitly skipped at
the user's instruction, matching the escape clause Section 8.4 already carried -- gating was done
by running the existing suites locally instead.

**What to do next:** nothing is scoped or pending in this plan. Both repos' full local suites are
green (core: 103 passed/1 skipped; frontend: 15/15 gradcheck scripts, 24/24 caseSpec+runner,
51/51 physics). Nothing has been committed in either repo as part of Step J -- both working trees
carry uncommitted changes; committing (and, in `warpSPH`, standing up the `.github/workflows/`
Section 8.4 deferred) are the two loose ends if a future session wants to close them out.

**Step H's highest-value finding has already been fixed** (in the frontend, `d9ad712`): the
Kolmogorov forcing's host-side scipy interpolator, 66-68% of host time above 57k particles, is now a
torch-native device-resident interpolator, worth 1.76x at 57k and 2.81x at 155k on that case. With
it gone, dispatch is the largest host bucket again below ~57k, which is what Steps I/J sit on top
of. Re-measured after the fix; the report's tables carry both the before and after.

**Note on GPU-sharing decisions (2026-08-17, from the user):** judge whether the GPU is free by
**percent VRAM free and utilisation**, not by whether some other process appears in `nvidia-smi`.
This box has 96 GB of VRAM, so an unrelated simulation holding a few GB is not a reason to defer a
test run. Step H's sweeps are the one place where actual contention matters, since they want the
whole device to produce meaningful timings.

**The one habit to carry forward, stated plainly because it paid off repeatedly:** every step from
C onward had at least one real, silent-corruption-class bug in the *written* sketch -- not
typos, not edge cases, but wrong results a casual read would not catch (Step C: a
`save_for_backward`/`backward()` alignment gap; Step D: a cache that would silently never hit;
Step E: two independent gradient-doubling bugs; Step F: a hazard serious enough that it was
checked *before* any implementation code existed, plus two more bugs in the "cheap wins" folded in
alongside it). None of these were caught by reading the plan carefully -- they were caught by
writing a targeted empirical check (a raw warp repro, a cross-check against an env-var-disabled
reference, a monkeypatch proving a test isn't vacuous) *before* trusting a design or *after*
implementing it but before calling it done. Budget for that on Step I too, especially anywhere it
touches the autograd bridge or introduces new shared/cached state -- the failure mode here has
consistently been "looks right, runs, produces a plausible number" with the actual bug only
visible in a gradient value or an object-identity check, never in a stack trace.

**Escape hatches available if something downstream looks wrong:** `WARPSPHCORE_DISABLE_FIELD_CACHE=1`
(kills the Field cache entirely, Steps B-F), `WARPSPHCORE_FIELD_CACHE_GRAD=0` (grad-path view
reuse only, Step E), `WARPSPHCORE_NULL_FILL=sentinel` (poisons disabled-correction reads instead
of zero-filling, Step C), `WARPSPHCORE_PROFILING=1` (restores real `record_function` hooks, Step
F), `WARPSPHCORE_DISABLE_BUNDLE=1` (falls back to per-call struct construction, added in Step H as
a *measurement* hatch -- see that step's finding 7). All five are independent single-variable
bisects, not reverts.

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

## Step C -- Null fields replace dummy tensors in `arg_extract.py`  [Design: Option A, 2026-08-17]

**Design Decision:** `flat_tensors` list will contain both `torch.Tensor` and `Field` objects.
StateAwareWarpFunction.forward branches on `isinstance(item, Field)`: Fields go directly to
`.view()`, Tensors go through `acquireView(t.detach())` for revalidation. This is correct
because null fields never change (standalone, owned), while caller tensors might be modified
in-place and need revalidation.

**Implementation:**

1. **Replace `getCachedDummyTensor` calls with `nullField` in `arg_extract.py` (lines 69-71, 102, 119, 126, 139-142, 170-179, 180-182):**
   ```python
   # OLD (lines 69-71):
   _d1f   = getCachedDummyTensor((1,),          dtype=torch_t, device=device)
   _d1Df  = getCachedDummyTensor((1, dim),      dtype=torch_t, device=device)
   _d1DDf = getCachedDummyTensor((1, dim, dim), dtype=torch_t, device=device)
   
   # NEW:
   _d1f   = nullField(FieldKind.SCALAR, dim, device, dtype=torch_t)
   _d1Df  = nullField(FieldKind.VECTOR, dim, device, dtype=torch_t)
   _d1DDf = nullField(FieldKind.MATRIX, dim, device, dtype=torch_t)
   ```
   All subsequent uses of `_d1f`, `_d1Df`, `_d1DDf` are placed directly into `flat_tensors`
   list (lines 227-242) -- Field objects mixed with Tensor objects. The flat list is now
   heterogeneous but remains deterministically ordered.

2. ~~Replace `getCachedDummyTensor` calls in `arg_check.py`~~ **-- skipped, see Status section's
   correction #1.** `checkInputRenormalization`/`checkInputGradHTerms`/`checkInputVolume`/
   `checkInputCRK`/`checkQV` have zero call sites in the current codebase; `arg_extract.py`
   never calls them and does its own inline None-handling. Left as unreachable dead code rather
   than converted, since there is no hot path to speed up here.

3. **Update `StateAwareWarpFunction.forward` (stateAwareWarpFunction.py, lines 56-60) to handle mixed list:**
   ```python
   with record_function("SAWF.forward - convert"):
       warp_arrays = []
       for item in flat_tensors:
           if isinstance(item, Field):
               wa = item.view(Role.PRIMAL)  # Standalone field, use directly
           else:
               wa = getCachedWarpArray(item.detach())  # Tensor, acquire/validate
           wa.requires_grad = item.requires_grad if isinstance(item, torch.Tensor) else False
           warp_arrays.append(wa)
   ```
   (Null fields never require grad; only caller tensors do.)

* **Gate:** operation matrix (full sweep) bit-identical; gradcheck green; bench shows
  the null-path improvement (dummy-tensor cost ~60 us → negligible dict lookup).
* **Completion (revised):** `getCachedDummyTensor` is **not** dead after Step C --
  `coreOperations/wp_interpolate.py:182-183` is a real, separate call site (see the Status
  section's correction #2). It stays in `cache.py`; only `arg_extract.py`'s own dummy
  construction was replaced.

## Step D -- View reuse, no-grad path only  [signed off, Section 3.3; Option A gating] -- ✅ DONE

**As actually implemented** (the sketch below, written before the step was built, had a bug --
see the correctness note under "Status as of 2026-08-17" above: `item.detach()` is a fresh Python
object on every call, so caching against it would never hit. Fixed by threading the caller's
original tensor object through instead and detaching only at the point of conversion):

1. **`src/warpSPHCore/autograd/cache.py`:**
   ```python
   def getCachedWarpArray(t: torch.Tensor, use_cache: bool = False) -> "wp.array":
       # t must be the caller's original tensor object, not an already-detached
       # copy -- see the docstring in cache.py for why.
       if use_cache:
           return acquireView(t)
       return castTorchToWarpAsBuiltins(t.detach().contiguous())
   ```
   (`use_cache` defaults `False`, not `True` as first sketched -- `stateLessWarpFunction.py`'s
   flat-tensor wrapper also calls this function and must keep today's always-fresh semantics
   unless it explicitly opts in, which it does not.)

2. **`StateAwareWarpFunction.forward` (`stateAwareWarpFunction.py`):**
   ```python
   is_tensor = tuple(isinstance(t, torch.Tensor) for t in flat_tensors)
   ctx.is_tensor = is_tensor
   ctx.any_requires_grad = any(t.requires_grad for t, tt in zip(flat_tensors, is_tensor) if tt)
   use_cached_views = not ctx.any_requires_grad

   warp_arrays = []
   for t, tt in zip(flat_tensors, is_tensor):
       if tt:
           wa = getCachedWarpArray(t, use_cache=use_cached_views)  # t, not t.detach()
           wa.requires_grad = t.requires_grad
       else:
           wa = t.view()  # Field entry
           wa.requires_grad = False
       warp_arrays.append(wa)
   ```

3. **`WARPSPHCORE_FIELD_CACHE_GRAD` was not added at this step** -- see the correctness/decision
   note above. It has no effect until Step E's grad-path caching exists for it to gate.

* **Gate:** operation matrix 2D/float32 smoke sweep clean (258 OK, 0 HIGH/ERR/NAN); full pytest
  suite (CPU+CUDA) green; all 13 gradcheck scripts green both standalone and via
  `test_gradcheck_scripts.py`; new `tests/operations/test_view_reuse.py` covers reentrancy tests
  1-3; bench (`docs/regression/bench_call_overhead_step_d.md`) shows dim=2 no-grad convert at
  24-25 us, down from Step C's 62-64 us and baseline's 291.8-319.8 us.

## Step E -- View reuse on the grad path  [Critical: zero-on-acquire contract] -- ✅ DONE

**Hazard (Section 3.3):** The previous `data_ptr`-keyed cache was deleted because it reused
`wp.array.grad` buffers without zeroing, causing non-reentrant backward and silent gradient
accumulation. This step reintroduces wrapper reuse but discharges the bug by design:
owned identity (Field attached to tensor, not inferred from storage address) + zero-on-acquire
contract + twice-in-process gradcheck gate.

**As actually implemented** (the sketch originally written here had the *same* `item.detach()`
identity bug already caught and fixed in Step D -- caching against a freshly-detached object never
hits -- and, more importantly, said nothing about two other bugs the zero-on-acquire contract
alone does not cover, both found and fixed while landing this step; see below):

1. **`src/warpSPHCore/autograd/stateAwareWarpFunction.py`:**
   ```python
   def _field_cache_grad_enabled() -> bool:
       return os.environ.get("WARPSPHCORE_FIELD_CACHE_GRAD", "1") != "0"

   # forward():
   use_cached_views = (not ctx.any_requires_grad) or _field_cache_grad_enabled()
   for t, tt in zip(flat_tensors, is_tensor):
       if tt:
           wa = getCachedWarpArray(t, use_cache=use_cached_views)  # t, not t.detach()
           wa.requires_grad = t.requires_grad
           if t.requires_grad and wa.grad is not None:
               wa.grad.zero_()  # zero-on-acquire, unconditional
       else:
           wa = t.view()
           wa.requires_grad = False
       warp_arrays.append(wa)
   ```
   `tape.zero()` at the end of `backward()` stays as belt-and-braces, unchanged.

2. **Bug found #1 -- grad-buffer double-counting when one tensor fills two roles.** The common
   case `referenceParticles=None` makes `qPos`/`rPos` (and `qSup`/`rSup`, `qMas`/`rMas`) the *same*
   tensor object. Under caching they now map to the *same* wp.array, so warp's adjoint kernels
   correctly sum both roles' contributions into that one shared `.grad` buffer -- but
   `backward()` was reading that buffer once per flat-tensor *position* (two reads of the same,
   already-complete total), and PyTorch sums whatever a Function's backward returns across every
   position a leaf occupies, so the total got doubled again on top. Fixed in `backward()` by
   deduplicating on `id(wa)`: only the first flat-tensor position referencing a given cached
   wp.array reports its gradient; later positions sharing it report `None`. (When caching is off,
   aliased positions build distinct wp.array objects with independently-correct partial `.grad`
   buffers, so `id(wa)` never collides and this is a no-op then -- the fix does not depend on
   caching being active.) Caught by `gradcheck_density_native.py` failing with analytical exactly
   2x numerical.
3. **Bug found #2 -- a latent `detach()` gap in `fieldRegistry.acquireView`'s two fallback
   branches** (the `WARPSPHCORE_DISABLE_FIELD_CACHE` escape hatch, and the non-contiguous-tensor
   path): both built from the bare tensor `t` instead of `t.detach()`, unlike the main cached-view
   path. Harmless through Steps B-D, since those branches only ever saw non-grad tensors (null
   fields; Step D's no-grad-only gate). Step E is the first caller that can hand `acquireView` a
   `requires_grad=True` tensor, and building off an undetached tensor let that conversion sit in
   torch's own autograd graph in addition to warp's tape -- doubling the reported gradient again,
   independently of bug #1. Fixed by adding `.detach()` in both branches. Caught by cross-checking
   a cached run against a `WARPSPHCORE_DISABLE_FIELD_CACHE=1` run that disagreed by exactly 2x.
4. **New regression gate: `scripts/gradcheck_twice_in_process.py`.** Neither bug was reachable by
   any existing test: `tests/operations/test_gradcheck_scripts.py` runs each gradcheck script in
   its own subprocess (isolated, so no cross-call state survives), and a single
   `torch.autograd.gradcheck` call, while it does call backward many times, always does so against
   *one* forward pass's tape/ctx -- it never exercises two independent `forward`+`backward`
   invocations sharing the process-level Field cache. The new script runs (a) the same leaf
   tensors through `torch.autograd.gradcheck` twice back-to-back, and (b) every
   `gradcheck_*_native.py` script's `main()` twice back-to-back in one process (fresh tensors each
   time, exercising the cache's handling of new objects arriving after now-dead ones). **Added to
   `tests/operations/test_gradcheck_scripts.py`'s `GRADCHECK_SCRIPTS` list**, so it now runs on
   every CI push (~13s locally).

* **Gate:** all 12 `gradcheck_*_native.py` scripts green (standalone and via
  `test_gradcheck_scripts.py`, now 14 entries); `gradcheck_twice_in_process.py` green (both
  scenarios); full pytest suite (90 tests, CPU+CUDA) green; 2D/float32 operation-matrix smoke sweep
  clean (258 OK, 0 HIGH/ERR/NAN).
* **Escape hatch:** `WARPSPHCORE_FIELD_CACHE_GRAD=0` forces fresh builds on the grad path only
  (no-grad still uses cache). For bisecting a suspected gradient bug, this is one environment
  variable rather than a full revert.
* **Performance:** `docs/regression/bench_call_overhead_step_e.md`. dim=2 grad-path convert:
  542.7-596.5 us (baseline) -> 133.8-145.8 us (Step D, uncached by design) -> 38.6-38.9 us (Step E,
  ~14x over baseline); total 1681.9-1995.4 us -> 346.4-414.5 us (~4.6-5.1x). Grad-path convert is
  now within ~1.6x of the no-grad path's ~24 us rather than ~6x higher.

## Step F -- `StateBundle` replaces the per-call closure  [Struct assembly 66.5us → ~10-12us] -- ✅ DONE

**Current Cost (Section 1.2):** `build_fn` closure performs ~51 warp struct field writes (66.5 us)
and is rebuilt from scratch every kernel launch from inputs that almost never change.

**The hazard this design has to defend against, found *before* writing implementation code.**
Given Steps C-E's track record, the plan's core assumption -- a persistent, mutable struct
instance, refreshed in place and shared across calls via `bundle.refresh(...)` -- was checked
directly against warp 1.16.0 first. Minimal repro: build one mutable `@wp.struct` instance, launch
a kernel with it under a `wp.Tape`, then **reassign the struct's array field to a different array**
before calling `tape.backward()` on that first tape. Result: the *second* array's gradient came out
populated (even though it was never used to compute the first tape's output), and the *first*
array's gradient came out zero. **`wp.Tape` does not snapshot a struct's field values at launch
time -- it holds a live reference to the mutable struct object and re-reads its fields lazily, at
`backward()` time.**

This means unconditional bundle sharing (as sketched) would silently corrupt an earlier
grad-requiring call's gradient any time its `.backward()` is deferred past a *later* call that
refreshes the same bundle -- which is ordinary PyTorch usage (build a graph across several
operators, call `.backward()` once at the end), not a contrived edge case. So: **`StateBundle`
reuse is gated to `not ctx.any_requires_grad`**, the same restriction as Step D's view-reuse cache,
but for a stronger reason and with no escape hatch -- there is no zero-on-acquire-style contract
that would make grad-path struct sharing safe (the corruption is a wrong *pointer*, not an
unzeroed accumulator), so unlike `WARPSPHCORE_FIELD_CACHE_GRAD` there is nothing to bisect against.
The grad path always gets a fresh, call-local struct set -- `arg_extract.py`'s original per-call
construction, entirely untouched.

**As actually implemented** (simpler than the sketch once the above is settled: since only `dim`
determines struct *type* today -- `structFor`'s table maps both `ExecutionMode.NONE` and
`REVERSE` to the same classes -- the "signature" collapses to `dim` alone; no LRU is needed since
at most 3 bundles (dim ∈ {1,2,3}) will ever exist):

1. **New file `src/warpSPHCore/autograd/stateBundle.py`:** `StateBundle` holds one instance each of
   `particleDataSoA_N` (×2, query/reference), `correctionData_N`, `adjacencyData`, `gridData`,
   `domainData`, `kernelState`, built via `structFor` (Requirement 3, closing the last
   `dim==1/2/3` ternary in `arg_extract.py` too). `refresh(wa, cfg)` compares each array-typed
   field's incoming `wp.array` against the previous call's (a single `self._last_wa` list, index-
   compared -- not 20+ named `_xxx_id` attributes) and writes only where it changed; scalar fields
   (`grid.hCell`, the four correction `use*` flags, all seven `kernelProperties` fields) are always
   assigned, since several legitimately change every call. `getStateBundle(dim)` is a plain dict,
   no eviction. No `torch.autograd` import anywhere in the file (Step I prerequisite, verified).
2. **`arg_extract.py`'s `build_fn` gained a `use_bundle: bool = False` parameter** rather than being
   replaced by a bundle-returning signature: when `True` it calls `getStateBundle(dim).refresh(wa,
   cfg)` and returns the bundle's structs directly; when `False` (the default, and every call site
   that predates this step) it does exactly what it always did. `cfg` was already a closure
   variable `build_fn` captures -- no new return value needed from `extractStateInfo`.
   `wrapper.py`'s own `build_fn` wrapper (which also splices in `additionalArguments`) just forwards
   the flag to the inner `state_build_fn`; that splicing logic is unaffected either way.
3. **`StateAwareWarpFunction.forward`** calls `build_fn(warp_arrays, use_bundle=not
   ctx.any_requires_grad)` -- one extra keyword argument, no new positional parameter, no
   `_N_NON_TENSOR` change.
4. **Section 3.7's two "cheap wins", folded in, each surfacing its own bug:**
   - **`record_function` gating.** New `warpSPHCore/profiling.py`: `WARPSPHCORE_PROFILING=1` gets
     the real `torch.profiler.record_function`; otherwise every call site gets
     `contextlib.nullcontext`. Applied at all ~19 `autograd/` call sites plus 17 more across
     `coreOperations/`, `radiusSearch/`, `crk/`, `pinv/`, `util/wp_util.py`, `renorm.py`.
     **Bug found: a circular import.** First placed at `autograd/profiling.py`; `util/wp_util.py`
     importing `from ..autograd.profiling import record_function` forced
     `warpSPHCore.autograd`'s `__init__.py` to run (which imports `arg_extract.py`, which imports
     `radiusSearch`), and that reentered `util` *while `util/__init__.py` was still on its own
     first import line* -- Python's reentrant-import handling returns the partially-initialized
     module, silently missing `castTorchToWarp` and everything else `util/__init__.py` hadn't
     reached yet. Every operator that builds adjacency failed with `NameError: name
     'castTorchToWarp' is not defined`, several calls deep, nothing to do with the actual change.
     Fixed by moving the gate to a genuinely zero-dependency top-level module
     (`warpSPHCore/profiling.py`) imported as the *very first statement* in
     `warpSPHCore/__init__.py` -- before even `type_config` -- so it is always fully populated in
     `sys.modules` before any reentrant chain elsewhere in the package can start.
   - **`allocateTorchWarp` dtype/device memoization.** Cache `(dtype, device) ->
     (torch_dtype, torch_device, trailing_shape, scalar_dtype)`, keyed per call site instead of
     re-resolved every time. **Bug found: `wp.Device` instances are unhashable** (define `__eq__`
     without `__hash__`), and `allocateTorchWarp` is sometimes called with one directly (e.g.
     `warp_array.device`) rather than a plain device string -- `TypeError: unhashable type:
     'Device'` on the very first adjacency-building call. Fixed by keying on `str(device)`.
* **New regression test:** `tests/operations/test_state_bundle.py`. One test confirms the no-grad
  path actually reuses the bundle (the entire point of the step); the other,
  `test_deferred_backward_across_two_grad_calls_not_corrupted`, pins the correctness property
  above directly -- two independent grad-requiring forward calls sharing `dim` (so they *would*
  share a bundle if the gate were ever weakened), neither backward run until both forwards
  complete, checked against a fully-sequential reference (which cannot suffer this class of bug).
  Verified the test is not vacuous by temporarily monkeypatching `use_bundle=True`
  unconditionally and confirming it fails with exactly the predicted symptom (one call's gradient
  reading as zero).
* **Gate:** full pytest suite (92 tests, CPU+CUDA) green; all 12 `gradcheck_*_native.py` scripts
  green; `gradcheck_twice_in_process.py` green; 2D/float32 operation-matrix smoke sweep clean (258
  OK, 0 HIGH/ERR/NAN).
* **Performance:** `docs/regression/bench_call_overhead_step_f.md`. dim=2 no-grad `build_fn`:
  142.1-152.7 us (baseline) -> 58.0-62.1 us (Step E) -> 10.3-12.1 us (Step F, ~13-14x over baseline);
  total 1197-1413 us (baseline) -> 219-316 us (Step F, ~4.5-5.5x), measured with
  `WARPSPHCORE_PROFILING=0` (the new default) -- `record_function`'s own overhead when actually
  hooked to a profiler turned out to be ~70-115 us per call on top of that, confirming Section
  3.7's second claim independently of the struct-assembly win.
* **Prerequisite for Step I satisfied:** no `torch.autograd` import anywhere in `stateBundle.py`
  (Section 3.6, requirement 6) -- verified by grep, not just by construction.

## Step G -- Forward-mode readiness audit (no forward mode implemented) -- ✅ DONE

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
  Requirement 4's *API-surface* half is Step I's job -- G audits the internals,
  I exposes them.

**Findings (2026-08-17).** Full write-up is in `warpier_core.md`'s new "Phase 6 -- Status:
readiness audited" block (that was this step's deliverable); the short version, and the two
places this document was wrong:

1. **Item 1 did NOT hold as landed, and it failed in a cache-warmth-dependent way.**
   `getStateBundle(dim, FORWARD)` validated the mode only implicitly, through
   `StateBundle.__init__` -> `structFor` -- which runs on a *miss*. Once any bundle existed
   for that dim, the FORWARD request hit the dict first and got the REVERSE-shaped bundle
   back with no error at all. `stateBundle.py`'s own comment asserted the opposite
   ("ExecutionMode.FORWARD raises in structFor() rather than silently handing back a
   REVERSE-shaped struct, so this does not need to guard against it separately") -- true of
   the cold path only, and the comment is why nobody looked. Fixed by validating ahead of
   the lookup; `test_forward_mode_rejected_regardless_of_cache_warmth` pins both cache
   states. Inert today (only `arg_extract.py` calls it, always with REVERSE), but it is the
   first guard Phase 6 leans on.
2. **Items 2 and 3 hold, with one caveat each.** No Python-side struct ternary survives
   outside `structFor`. But `util/stateUtil.py` has three concretely-typed `@wp.func`
   overload sets per dim, dispatched by *warp's* overload resolution on struct type -- so
   requirement 3's "Phase 6 registers `particleDataSoA_2_dual` into the same table" is not
   the whole change if dual structs ever happen; each needs its own overload set too. And
   `stateBundle.py` itself imports no torch (verified by constructing a bundle with
   `torch.autograd.Function` replaced by a poison class), but it *lives* in the `autograd/`
   package, so importing it runs that `__init__.py` and drags `StateAwareWarpFunction` in
   anyway. Requirement 6 is real at the code level, cosmetic at the import level; moving the
   file out of `autograd/` is folded into Step I, which reworks these imports regardless.
3. **Item 4's spike passes, and Tier 1 is cheaper than requirement 3 assumed:** 14/14 cases
   (Interpolate, Gradient x 4 schemes, Divergence, Curl, Laplacian; 1D and 2D; scalar and
   vector fields; renormalisation off and on) agree with the reverse-mode Jacobian
   contracted with the tangent, to float64 round-off (7e-17 … 2.7e-16), with `f(0) == 0`
   exactly -- linear, not merely affine. **Tier 1 needs no new struct type**, so it needs
   no `FORWARD` row in `structFor` and no widening of the bundle key: it is the existing
   kernel on the existing structs with the tangent array substituted for the value array.
4. **This document's Step G text named the wrong reference and would have validated a
   broken bridge.** `torch.autograd.functional.jvp` does not raise here -- it returns a
   **silently zero** tangent (rel_err 1.0 against the verified answer), because its
   double-backward trick needs a differentiable backward and `strict=False` degrades to
   zeros instead of erroring (`strict=True` does raise). A Phase 6 bridge that also emitted
   zeros would agree with it perfectly. The spike uses `torch.autograd.functional.jacobian`
   contracted with the tangent instead -- reverse-mode only, so it needs nothing the
   gradcheck suite has not already validated, and it is exact on cases this small. The
   honest path, `torch.autograd.forward_ad`, raises `NotImplementedError: You must implement
   the jvp function for custom autograd.Function` -- which names Phase 6's actual deliverable.
5. **A real bug in shipped code, unrelated to Steps A-F, found by the spike:**
   `computeRenormalizationMatrices` mutated the *caller's* `OperationProperties`
   (`operationProperties.operation = WarpOperation.Covariance`). A caller that reused that
   object afterwards silently launched Covariance where it asked for a Gradient and got a
   plausible `(N, D, D)` tensor back. No call site in either repo hit it -- they all
   construct a fresh properties object inline -- but **Section 3.5's own suggested
   follow-up (hoist those constructions out of the hot path so a reusable, hashable
   properties object can key the `StateBundle`) would have introduced this bug rather than
   found it.** Fixed with `dataclasses.replace` in `renorm.py`; pinned by
   `tests/operations/test_renorm_no_caller_mutation.py`, verified non-vacuous by
   temporarily restoring the mutation (4/4 fail). Also confirmed that line was the *only*
   in-place mutation of an `OperationProperties` field anywhere in either repo, so
   Section 3.5's `frozen=True` proposal is now unblocked -- worth taking at Step I, where
   the bundle key wants it.
* **Spike kept, not thrown away.** `scripts/spike_forward_mode_tier1.py` is wired into
  `tests/operations/test_gradcheck_scripts.py` (as `SPIKE_SCRIPTS`, distinct from
  `GRADCHECK_SCRIPTS` since it checks a forward JVP identity, not a backward pass), ~3s.
  What it pins is load-bearing for Phase 6's entire cost estimate -- exact linearity in the
  field values -- and if a future kernel change breaks that, nothing else in the suite
  would notice.
* **Verification:** full pytest suite including CUDA-parametrized tests, **112 passed**; all
  14 gradcheck scripts plus the spike via `test_gradcheck_scripts.py` (15 passed, CPU by
  construction); 2D/float32 operation-matrix sweep on both devices clean (**258 OK, 0
  HIGH/ERR/NAN**), matching Steps D-F's gate exactly. A jittered variant
  (`--jitter 0.2`) reports 74 OK / 184 HIGH -- **checked against unmodified `HEAD` and
  bit-identical there (74/184)**, so those cells are a pre-existing property of the
  jittered configuration (which is a diagnostic, not the `--ci` gate), not anything this
  step did. Worth knowing before someone runs the jittered sweep and reads it as a
  regression.

## Step H -- Real-world bottleneck audit across particle-count scales  [inserted 2026-08-17]

**Why this step exists.** A real ~70k-particle, no-grad simulation went from 45 to 42 minutes
after Steps A-F landed -- about 7%, against a per-call microbenchmark showing the fixed dispatch
overhead these steps target dropped ~4-6x (`bench_call_overhead_step_f.md`). Both numbers are
correct; they answer different questions. `bench_call_overhead.py` isolates exactly the thing
Steps A-F changed -- repeated calls to one operator (Density), on a fixed particle count, against
**a `CompactHashMap` built once and reused across all 200 timed iterations** (the script's own
comment: *"measures operator call overhead, not hash-map construction"*) -- and confirms that
piece works, with `torch.cuda.synchronize()` around the timed region so the numbers are real
completed-work time, not queued-but-unfinished dispatch. What it cannot tell you is what fraction
of a real simulation's wall-clock time that piece actually is, because a real run also pays for
neighbor-list rebuilds every step (excluded from the bench by construction), genuine GPU kernel
compute scaling with N and neighbor count (out of scope for this plan by design -- "must not touch
kernel math"), the integrator's own tensor churn (Section 7, ~40 allocations/RK4 step, explicitly
deferred as "after this plan"), and whichever CPU-GPU synchronization points a real solver's
control flow introduces (adaptive timestepping, logging, I/O) that an isolated single-operator loop
never triggers. Section 1.1's own baseline data already shows the CPU-dispatch-bound / GPU-compute
-bound crossover sitting somewhere between N=20k and N=200k (Density: 919us at 2k, flat through
20k, 7484us at 200k) -- so at 70k, before any of Steps A-F landed, a meaningful and growing share
of each call's cost was *already* GPU compute, not the CPU marshalling this plan fixes. The
per-call multiplier is real and validated; whether it was ever the dominant cost of a full run at a
given scale is a separate, unanswered question this step exists to answer with data instead of
inference from a different benchmark.

**This is diagnostic, not a caching step -- treat it with the same rigor as Steps C-F's
verification, not as a lighter aside.** The output feeds a real decision: whether something Steps
A-F did *not* touch needs a deeper architectural change before Steps I/J move the interface under
it. Section 7 (integrator buffer pool) is the leading candidate already written up in this
document; this step's job is to find out whether it -- or something else entirely (neighbor-list
rebuild cost, most likely) -- is actually where the time goes, rather than assuming Section 7's own
"~3% drag" estimate (Section 7.2, itself derived from the *no-grad-only field-cache* miss rate, not
from a full-run profile) still holds now that the field cache and StateBundle it was compared
against have both landed.

**Methodology.**

1. **Real workload, not synthetic.** Use `~/dev/warpSPH` (the frontend), not
   `bench_call_overhead.py` or the gradcheck/operation-matrix scripts -- those exercise correctness
   and isolated per-call cost, not a full solver loop. Pick one or two representative case setups
   already in the frontend's `caseUtils`/examples (a weakly-compressible or similar production-
   shaped case, matching what actually produced the 45-minute run if that setup is available)
   rather than inventing a new one, so the profile reflects real usage.
2. **Particle-count sweep spanning the CPU-dispatch-bound to GPU-compute-bound crossover and
   beyond it**, informed by Section 1.1's baseline: something like 2k, 20k, 70k (the reported data
   point), 200k, and at least one point past 500k if it fits in the RTX PRO 6000's 96GB, in
   whichever of 2D/3D matches real usage (do both if both matter to the project -- 3D's per-particle
   neighbor count is higher, which changes the neighbor-rebuild-vs-compute balance). Run each scale
   long enough to profile a representative handful of steps (not the full run -- profiling overhead
   itself would distort a 45-minute measurement), but confirm the profiled steps are steady-state
   (post-warmup, no one-time kernel-compile artifacts of the kind seen in Steps C-F's CUDA bench
   sweeps skewing the sample).
3. **Profile with `WARPSPHCORE_PROFILING=1`** (Step F's gate; without it, the `record_function`
   regions Steps A-F's own bench relies on are no-ops and this step loses the ability to attribute
   time the same way those docs already do) plus `torch.profiler`'s CUDA activities, attributing
   wall-clock time per step to: neighbor-list/adjacency rebuild, operator dispatch overhead
   (extract/convert/build_fn/allocate -- what Steps A-F targeted, expected to now be small),
   GPU kernel compute proper, integrator tensor churn (the out-of-place update allocations Section
   7 describes), and CPU-GPU synchronization points (adaptive `dt` computation reading a value back
   to host is the most likely culprit -- check whether one exists in the profiled case and whether
   it forces a device sync every step). Exclude plotting/I/O from the profiled window, or account
   for it separately if it can't be disabled.
4. **One clean before/after comparison on the real workload**, not just synthetic-bench numbers:
   at one representative scale, run the profiled steps twice -- once as landed, once with every
   Steps A-F escape hatch flipped off (`WARPSPHCORE_DISABLE_FIELD_CACHE=1`,
   `WARPSPHCORE_FIELD_CACHE_GRAD=0`, and Step F's `use_bundle` gate would need a matching
   escape hatch added if one doesn't already exist -- note as a finding if it's missing) --
   to get an in-situ confirmation of what Steps A-F actually bought on this workload, separate from
   what remains.
5. **Compute cost of this step itself is non-trivial** (multiple real simulations at multiple
   scales, some potentially large, some possibly running many minutes each). Confirm the GPU is
   actually free before each run rather than assuming the green light from Steps A-F's CUDA bench
   work still stands by the time this step executes -- this machine's GPU is shared with other
   simulation work.

**Deliverable:** a written report (`docs/regression/real_workload_bottleneck_audit.md` or similar)
with a bottleneck-by-scale table -- which bucket dominates at which N, in which dimension -- and an
explicit recommendation, one of:

* **Nothing further needed before Steps I/J.** Steps A-F's target was never far from the true
  bottleneck at the scales that matter to this project; Section 7's estimate holds up under a real
  profile. Proceed to Steps I/J as planned.
* **Neighbor-list rebuild is the dominant or a major cost.** Investigate whether
  `AdjacencyList`'s existing (per its own docstring in `adjacency_t.py`) but seemingly-unused
  Verlet-style rebuild-threshold fields (`queryPositions`/`querySupports` etc., "extra neighbors to
  avoid rebuilding the neighbor list every step") are actually wired up anywhere, or are dead
  scaffolding from an earlier design. Scope a follow-on step to actually use a buffered rebuild
  before Steps I/J move the interface underneath whatever that fix touches.
* **Integrator tensor churn is the dominant or a major cost.** This promotes Section 7's buffer
  pool (`PooledState`/`WarpState`, the fused `axpy` kernel, CUDA-graph capture on the `NONE`-mode
  path) from "after this plan" to "before Steps I/J" -- Section 7.4 already ties its `ExecutionMode`
  gating to the same switch this plan introduced, so the interface dependency runs the other way:
  doing it first means Steps I/J's `SPHContext` can be designed as the pool's natural handle
  (Section 8.3 already anticipates this) rather than retrofitted onto it later.
* **GPU kernel compute time itself dominates at the scales that matter.** Out of scope for a
  quick follow-on -- this plan does not touch kernel math -- but worth stating plainly rather than
  implying more headroom exists than does. CUDA graph capture (a dispatch-level, not kernel-math,
  mitigation, and Section 7.3's territory) is the one remaining lever this plan's approach can still
  pull; note whether it looks worthwhile at the profiled scales.
* **Dispatch overhead is still non-trivial even after Steps A-F**, once real multi-operator
  sequencing (not an isolated single-operator loop) is accounted for. If so, instrument the actual
  cache hit rates (Field cache, StateBundle) in a real run rather than trusting Section 2.3's
  projected hit-rate, which was derived analytically, not measured against a real integrator loop.

* **Gate:** the report exists, names a scale-by-scale bottleneck, and gives one of the above
  recommendations (or a mix, by scale) with enough specificity that Steps I/J -- or a newly-scoped
  step ahead of them -- can be planned from it without re-deriving the profile.

### Step H results (2026-08-17) -- ✅ DONE

Report: `docs/regression/real_workload_bottleneck_audit.md`. Harness:
`scripts/bench_real_workload.py`. Workload: the frontend `dambreak` case as fully-periodic
Kolmogorov flow (the invocation actually used for data generation), 2.4k -> 155k particles, via
`warpSPH.runner.run` so the profile covers the real deltaSPH/RK2 loop.

**Verdict: "Nothing further needed before Steps I/J", plus one frontend fix worth far more than any
remaining core work, and two of this step's candidate follow-ons explicitly ruled out on evidence.**

1. **The 45 -> 42 min observation is fully accounted for; there is no missing regression.** In-situ
   before/after (every caching layer disabled via escape hatches, 40 measured steps) gives
   **1.04-1.16x on the full solver loop** -- a near-constant **~0.6-1.2 ms/step** saved at every
   scale. Against the scipy-era ~20 ms step at 57k that is ~5%, matching the reported ~7% to within
   either measurement's precision. The 4-6x from `bench_call_overhead.py` is also right: it measures
   the marshalling Steps A-F removed, while the loop still pays ~350 `cudaLaunchKernel`s and ~41
   host stalls per step, warp's launch machinery, and the Python chain through 16 operator wrappers.
   With the forcing now fixed the same ~1 ms is 8-12% of a step rather than ~5%, purely because the
   denominator shrank.
2. **The audit's headline finding was the Kolmogorov forcing at 66-68% of host time from 57k
   particles up -- and it has since been fixed in the frontend.**
   `caseUtils/weaklyCompressible.py`'s `forcing()` evaluated a **scipy
   `RegularGridInterpolator` on the host every stage**, round-tripping the position array
   device->host->device: 16.4 ms/step in the scipy call plus 9.1 ms in `aten::copy_` and 3.7 ms in
   `cudaStreamSynchronize` at 155k. `d9ad712` ("add torch interpolator") replaced it with a
   torch-native device-resident port. Re-measured with the same harness: the forcing bucket drops
   26.36 -> 2.27 ms/step at 57k and 35.13 -> 5.62 at 155k, i.e. **step time 20.56 -> 11.67 ms
   (1.76x) and 48.03 -> 17.11 ms (2.81x)**. It has left the scaling cost -- flat through 57k, and
   at 155k the whole step is now within 5% of the same case with the forcing removed entirely. It
   also gains differentiability the scipy version could not offer. **Confirmed on the real run:
   the ~70k production case went 42 -> 16 minutes (2.6x, user-reported), inside the bracket the
   harness predicted.** That is the second time a profiled 15-step window on this case has matched
   a full run's wall clock (the first being Steps A-F's own ~7%), which is worth trusting the
   harness for next time rather than re-deriving.
3. **Steps A-F targeted the right thing for the scales this project runs at.** Step time is nearly
   flat from 2.4k to 57k (8.71 -> 10.66 ms without forcing, a 1.2x rise for 24x the particles) and
   **dispatch is the largest single host bucket at every scale up to 57k, at a flat ~4.0-4.4
   ms/step** -- and with the forcing fixed it is the largest one again in the case as actually run
   (31-32%). The run only becomes genuinely GPU-bound at ~150k (61-67% device-busy). Section 1.1's
   predicted crossover is confirmed, just located higher than the 20k-200k bracket suggested.
4. **Section 7's integrator buffer pool is NOT the bottleneck in 2D -- do not promote it on this
   evidence, but do not dismiss it either.** `[Integration]` regions are 6-10% of host time and
   0.4-0.9 ms/step of device time in 2D; the integrator's own `aten::copy_` traffic is ~0.05
   ms/step, so Section 7.2's "~3% drag" estimate survives a real 2D profile. **This ruling-out is
   load-bearing:** the first, naive attribution said the opposite (see finding 6) and would have
   promoted this work on false evidence. **But the scenario probes (finding 9) show the integrator
   as the *largest* bucket under CRKSPH (43%) and in 3D (45%)** -- so Section 7 should be
   re-profiled against those before being scoped, not written off from the 2D numbers.
5. **Neighbour rebuild is not the bottleneck in 2D either, and the Verlet scaffolding is live, not
   dead.** 1.6-1.9 ms/step host, 10-15% of host time and *falling* with N. `[Verlet] Checking
   validity of prior neighborhood` runs ~3x per step and is doing its job -- answering this step's
   open question about `adjacency_t.py`'s rebuild-threshold fields. Same 3D caveat as above: 29% of
   host time there.
6. **Two attribution bugs in the harness, each of which reversed a conclusion before being caught.**
   Recorded because they are the same failure class every step from C onward produced -- plausible
   numbers, wrong answer. (a) `key_averages()` merges a key's CPU and CUDA aspects into one
   `DeviceType.CUDA` entry, so each `record_function` region carries the device time of every kernel
   inside it; summing that gave a **GPU share of 175%**, which is the only reason it was noticed.
   Fixed by aggregating raw events filtered on `is_user_annotation`, now matching the profiler's own
   "Self CUDA time total" exactly. (b) Bucketing host events by their own name charged
   `aten::copy_` to the *integrator*, putting 18 of 20 ms/step at 57k there and making Section 7
   look like the bottleneck; walking the parent chain showed 9 of those ms inside
   `[warpSPH] - computeForcing` and ~0.05 ms/step actually in `[Integration]`. A follow-on bug in
   that same fix charged a region's own self-time to its parent, hiding another 16.4 ms/step of
   forcing cost in the generic `solver` bucket.
7. **`WARPSPHCORE_DISABLE_BUNDLE` added** (this step's methodology anticipated it: "note as a
   finding if it's missing"). Step F ships no hatch by design -- the grad path is gated on
   `requires_grad`, not a variable, so there is nothing to bisect -- but without one the before/after
   comparison silently left the bundle on in both arms. It is a **measurement-only** hatch: turning
   it off only falls back to `arg_extract.py`'s original per-call struct construction, so it can
   never be the unsafe direction. Verified non-inert (`_BUNDLE_CACHE` stays empty with it set)
   rather than assumed.
8. **Frontend fix required to run the sweep at all, worth keeping on its own merits** (suggested by
   the user mid-step): `perlinNoise*D` needs its grid resolution to be an exact multiple of every
   octave's lattice frequency -- 16 for the default 4 octaves at base frequency 2 -- so `nx=39`
   died with a shape mismatch several frames deep and the sweep was pinned to multiples of 16.
   `warpSPH/math/noiseFunctions/generator.py` now generates at `paddedNoiseResolution(n, ...)` and
   resamples down to `n` via a wrap-aware separable linear resample (`resampleNoise`), keeping
   tileability across the seam. Already-valid resolutions take neither path and stay bit-identical.
   Frontend suite after the change: 87 passed, 1 skipped, 0 failed -- unchanged from the Steps A-F
   baseline.
9. **The harness is scenario-driven, and the bottleneck moves substantially between scenarios** --
   documented in the report's "Re-running with other scenarios" section and in the script's own
   docstring, at the user's request, so it can be pointed at other paths through the solver rather
   than re-derived. `--case` / `--preset` / `--param` / `--spec` cover case, scheme, integrator,
   kernel, support mode and dimension. Five verified probes, same core, same harness:
   dambreak+Kolmogorov RK2 2D is dispatch-bound (32%); `integrationScheme=rungeKutta4` doubles
   operator calls (16 -> 30) and 2.6x's allocations (409 -> 1060) at identical N; `--case tgv` is
   dispatch 47% / integrator 28%; `--case sod2d --spec scheme=CRKSPH` is **integrator 43%**; and
   `--case sod3d` at only 16k particles is **integrator 45% / neighbour 29% / dispatch 5%** -- the
   inverse of the 2D conclusion. Anything scoped off this audit should be re-profiled in the
   scenario it targets; that is the reason this is a harness and not a one-off script.

**Consequence for Steps I/J:** unblocked, and no new step is needed ahead of them. Their own
rationale (retiring the 27 hot-path dtype probes, the per-call `additionalArguments` re-analysis,
somewhere to put `ExecutionMode`) stands; the *performance* argument for them is now known to be
worth ~1 ms/step on this workload, not more.

**Host-stall removal, core side: done, and it beat every caching layer in this plan.** Step H's
sync census (counted in Python, not inferred from profiler regions -- the region numbers misread
this, see the report) found 39.1 device->host readbacks per step, of which two sites were core:
`_minimum_image_delta` testing `bool(periodicity[d].item())` once per axis per call (11.5/step,
reading a domain property that is *fixed for the whole run*), and `renorm.py`'s
`if torch.any(lowNbrMask):` guarding a masked assignment that is itself a synchronising op
(2.0/step). Both are now branch-free via `torch.where`. **39.1 -> 25.5 readbacks/step, and per-step
wall clock 10.64 -> 8.89 ms at 19k (1.20x), 11.07 -> 9.78 at 57k (1.13x), 16.38 -> 15.19 at 155k
(1.08x)** -- i.e. **eleven lines of core code are worth more than Steps A-F's entire caching
programme** (1.19-1.75 vs ~0.6-1.2 ms/step). Verified bit-identical before being trusted (all
periodicity patterns, motion to 3x the box length, seam crossing; renorm on values *and*
gradients with the masked path exercised), pinned by `tests/operations/test_no_host_sync.py`
(counts readbacks, not time; confirmed non-vacuous by restoring both old implementations), and
green across the core suite, 2D+3D operation matrix, all 15 gradcheck scripts and the frontend
suite.

**Host-stall removal, frontend side: also done** (`3d4646a`..`c3238c3`, by the user, same two
patterns across `math/__init__.py`, the three mDBC modules, `shifting/delta.py`, the deltaSPH and
dfsph schemes and `systems/weaklyCompressible.py`). Re-measured: **39.1 -> 25.5 -> 8.6 readbacks
per step, a 78% reduction**, with what remains either load-bearing (the Verlet rebuild decision at
2.9/step, the runner's NaN check at 1.0) or setup-only. Frontend suite still 104 passed, 1 skipped.

**Cumulative effect of Step H's three fixes on per-step wall clock** (same case, same harness):
19k particles 12.35 -> 8.25 ms (**1.50x**), 57k 20.56 -> 8.35 ms (**2.46x**), 155k 48.03 -> 13.49 ms
(**3.56x**). Two consequences worth carrying into Steps I/J: **step time is now essentially flat
from 2.4k to 57k** (7.91 -> 9.37 ms for 24x the particles -- 19k and 57k within 2% of each other),
so that range is almost purely fixed-overhead-bound; and **dispatch is now unambiguously the
largest host bucket at 35-37%** (flat ~4.1 ms/step) with blocking down from 19-24% to 11-15%. At
155k the run is 72.5% device-busy, i.e. genuinely GPU-bound, where no host-side work helps much.
Note what the dispatch figure does *not* imply: Steps A-F already removed the marshalling, and
disabling them costs only ~0.6-1.2 ms/step, so the residual ~4 ms is warp's launch machinery and
the Python call chain -- Steps I/J should be justified on ergonomics and forward-mode readiness,
with any further dispatch win measured rather than assumed.

**Correction to Section 7.3's CUDA-graph claim (measured 2026-08-17, 19,044 particles, 20 steady
steps -- see the report's "What actually blocks CUDA-graph capture").** Section 7.3 says capture is
gated on a "fixed step shape", and earlier drafts of the Step H report repeated that as "gated on
the integrator". Both are wrong. (a) The step shape is *already* stable in steady state: the one
genuinely data-dependent branch, the Verlet rebuild, **fired 0 times in 20 steps** (the validity
check runs 6x/step and reuses). (b) The real blockers are **49 `.item()` host readbacks per step**
(illegal inside a captured region -- capture forbids synchronization) and **12 `aten::nonzero` per
step** (output shape is data-dependent, so launch sizes are unknown at capture time), and they live
almost entirely in *frontend solver modules*, not in the integrator and not in this repo: 15/step in
the Verlet validity check, 8 in `computeForcing`, 6 in `[deltaSPH - 17] - enforce updates`, 3 in
`computeDeltaShift`, the rest scattered. (c) The integrator's allocation churn does not forbid
capture at all -- repeated identical allocation sequences are what a graph memory pool exists for --
but a captured graph bakes in device addresses, so **Section 7.3's `PooledState` is what makes
capture *worth* anything rather than what unblocks it.** The dependency runs the opposite way to
Section 7.3's wording.

## Steps I / J -- The interface break

Detailed in Section 8, and deliberately last -- now gated on Step H's findings too, not just on
Steps A-F being landed: land, measure, stabilise, *and profile the real workload* before the
interface they sit behind starts moving. Step I introduces
`OperatorSpec` / `SPHContext` / `launchOperator` and ports the core, with
`warpWrapper2` reduced to a shim so the frontend is untouched; Step J migrates the
frontend's 24 wrapper sites and 65 `warpOperation` sites and deletes the shim.

### Step I -- ✅ DONE (2026-08-18)

**Prerequisites first, per Step H's audit (line ~726 above):** `OperationProperties` is now
`@dataclass(frozen=True)` -- unblocked by Step G's finding that its one in-place-mutation site
(`computeRenormalizationMatrices`) was already fixed and confirmed the only such site in either
repo. `stateBundle.py` moved from `autograd/` to `util/` (next to `fieldRegistry.py`, which it
already depended on) -- it imports no torch and has nothing to do with the autograd bridge, but
living under `autograd/` meant importing it dragged `StateAwareWarpFunction` in via that package's
`__init__.py`. `ExecutionMode` gained an `AUTO` member (`SPHContext.mode`'s default) alongside
`NONE/REVERSE/FORWARD` -- rejected at the `SPHContext` boundary exactly like `FORWARD` is rejected
internally by `structFor`/`getStateBundle`; nothing branches on `NONE` vs `REVERSE` yet, so there
is no behaviour to resolve `AUTO` into beyond that guard, and `structFor`'s existing `KeyError`
fallback already rejects it cleanly if it ever reaches the struct table directly.

**Machinery:** `autograd/operator_spec.py` -- `OperatorSpec`, `OutputSpec`, `ExtraSpec`/`ExtraKind`,
`ShapeOf`, `ThreadSpec`, `Corrections`, `SPHContext`, `launchOperator`, matching Section 8.2's
sketch. `warpWrapper2`'s body (struct extraction + additionalArguments split +
`StateAwareWarpFunction.apply`) was extracted into a private `_launch` in `wrapper.py`; both
`warpWrapper2` (kept, now a real one-line shim) and `launchOperator` call it, so the two entry
points cannot drift apart -- there is exactly one place gradients or struct assembly could go
wrong, not two parallel implementations to keep in sync.

**All 9 real call sites ported** (Density, CRK volume, CRK density, CRK moments, Covariance,
Interpolate, Gradient, Divergence, Curl, Laplacian -- 10 kernels, 9 backend functions, `crk_moments`
is 7-output). `warpWrapper2` is no longer called anywhere in this repo's core; it exists solely for
the frontend's own call sites, which Step J migrates.

**Section 8.3 corrected against what the code actually does, twice:**
1. **"Output dtypes resolved at import" holds for 5 of 9 operators, not all of them.** Density,
   Covariance, and the three CRK operators have output dtypes that depend only on
   dim/precision/masses-dtype -- genuinely static, or as static as `scalar_t` (the configured
   precision type) already was. But Interpolate/Gradient/Divergence/Curl/Laplacian's output dtype
   packs the *runtime* shape of a `queryValues`/`referenceValues` tensor that is not known until
   call time -- no amount of declaring things earlier makes an unknown-until-called tensor shape a
   compile-time constant. `OutputSpec.dtype` accordingly accepts a resolver callable
   (`Callable[[ctx, extras], dtype]`) alongside a literal value; the win for those five operators is
   that the resolution logic is declared once next to the kernel instead of hand-inlined at each
   call site, not that the runtime probe disappears. Recorded in `operator_spec.py`'s own docstring,
   not just here.
2. **The resolver pattern itself changed mid-port.** Curl and Divergence's output shape depends on
   the *unflattened* `queryValues.shape[1:]` in a dim/mode-dependent way (Curl: different formula
   for `dim==3/2/1`; Divergence: `inputShape[1:]` vs `inputShape[:-1]` depending on `dotMode`) that
   is not recoverable from the flattened `(N, flatInputShape)` tensor a resolver actually receives.
   Rather than re-deriving each formula a second time inside the resolver (a second place to get it
   wrong), every one of the five vector-output operators' resolver just reads the already-computed
   `flatOutputShape` extra straight back out of the `extras` dict passed to `launchOperator` --
   faithful to the original by construction, immune to shape-formula bugs by not re-implementing the
   formula at all.

**A real bug caught immediately, the first time `OutputSpec.resolve` ran against a *literal* dtype
instead of a resolver:** the initial implementation used bare `callable(self.dtype)` to tell a
resolver function apart from a literal dtype. `scalar_t` (and `wp.int32`, and every other Warp
dtype) is a *class*, and classes are callable -- `callable(scalar_t)` is `True` -- so CRK volume's
static `outputDtypes=scalar_t` port crashed immediately (`TypeError: scalar_base.__init__() takes
from 1 to 2 positional arguments but 3 were given`), caught by `pytest -k crk` on the very next
call site ported, before the CRK-family port was committed. Fixed by discriminating on
`types.FunctionType` instead of bare `callable()` -- a resolver is always a plain function/lambda
declared alongside the `OperatorSpec`, never a type.

**Verification:** full pytest suite (118 passed, 1 skipped) after every call site ported, not just
at the end. `operation_matrix.py`'s CI-gated sweeps clean on this machine: 2D/float32 (258 OK),
1D/float32 (282 OK), 2D/float64 (258 OK), 3D/float32-CUDA (258 OK), 0 HIGH/ERR/NAN in every one; the
`--jitter 0.2` diagnostic sweep reproduces the documented pre-existing 74 OK/184 HIGH exactly (not
a regression -- matches the number Step G's write-up already pinned against unmodified `HEAD`).
Frontend (`~/dev/warpSPH`, editable install, so it picked up every change with no reinstall):
`tests/test_gradcheck_scripts.py` 15/15 passed; `tests/test_physics.py` +
`tests/test_caseSpec.py` + `tests/test_runner.py` all green (exit 0, no failures in the dot output).

**Performance check, and a hazard worth recording for whoever runs this next:** `nvidia-smi`
reports this machine's GPU as an RTX 3090 (24 GiB); every `docs/regression/bench_call_overhead_step_*.md`
number recorded during Steps C-F was measured on a **different card** (an RTX PRO 6000
Blackwell, per those docs' own header lines) -- comparing this session's ~550-650 us readings
against those documents' ~150-320 us numbers directly would have read as a 2-4x regression that
isn't real, just different hardware. Verified properly instead: `git worktree add` a throwaway
checkout of the pre-Step-I commit (`a5ccf1b`), ran the identical bench there with `PYTHONPATH`
pointed at the worktree's `src/` (not `pip install -e .`, which rewrites this environment's global
editable-install pointer and would have left every subsequent command in this same session
importing from the wrong tree -- caught and undone immediately after it happened once). Pre-Step-I
(518/625/574 us) and post-Step-I (545/642/622/640 us) fully overlap at dim=2, N=19881, no-grad --
no measurable regression, consistent with `launchOperator`'s added work (a `dict` build, a handful
of `isinstance` checks, a tuple comprehension) being negligible next to the ~550+ us this bench's
"other" bucket already attributes to kernel dispatch and Warp/Tape overhead on this card.

**What Step I is not:** no new caching layer, no change to which struct types get built or what
gets zeroed on the grad path -- `_launch` is untouched Steps A-H logic, just reachable from two
entry points now instead of one. `ExecutionMode.AUTO` is declared and rejects `FORWARD`, but
resolves to nothing else yet; there is no behaviour today for a resolved `NONE` vs `REVERSE` to
attach to; this is Section 8.2's "home for `ExecutionMode`" being built, not Phase 6 starting early.

**What's left:** Step J -- migrate the frontend's 24 `warpWrapper2` call sites and 65
`warpOperation` sites onto `SPHContext`/`launchOperator`, module by module, gated on the frontend's
own gradcheck and physics suites staying green; then delete the `warpWrapper2` shim and the
`defaultStateArguments` tuple path. Section 8.4's prerequisite for Step J (`warpSPH` CI running its
existing `tests/`) has not been done -- worth checking whether it still holds before starting, not
assuming it from this paragraph.

### Step J -- ✅ DONE (2026-08-18), with two corrections

**CI prerequisite deliberately skipped, at the user's explicit instruction this session.** Section
8.4's "give `warpSPH` a `.github/workflows/`" prerequisite already carried its own escape clause
("skip this step as the tests are computationally expensive... especially before this major rework
is done") -- the user re-confirmed taking that branch rather than standing it up. All gating for
this step was therefore done by running the frontend's existing suites locally, by hand, at each
checkpoint (below), not by CI.

**Correction 1 -- the 65 `warpOperation` call sites needed no changes at all.** `warpOperation`
(`operations.py`) dispatches straight to `_computeSPH*_stateBackend`, and Step I already ported
every one of those backends (Density, Interpolate, Gradient, Divergence, Curl, Laplacian, Covariance,
the three CRK operators) onto `launchOperator`. The 65 frontend call sites were already running
through the new ABI the moment Step I landed; there was nothing left for Step J to touch there. The
plan's phrasing ("migrate... the 65 `warpOperation` sites") assumed they needed the same treatment as
the `warpWrapper2` sites -- they didn't, because they sit one layer higher than the ABI boundary Step
I actually moved.

**The real work: all 24 `warpWrapper2` call sites** (`modules/{liu,dissipation×3,util×2,pressure,crk×2,
surfaceDetection×3,mdbc,shockCapturing×2,deltaSPH×2,compSPH×3,incompressible,adaptiveSupport×2}`,
`sample/wp_deltaShift.py`) now declare a module-level `OperatorSpec` next to their kernel and call
`launchOperator(spec, SPHContext(...), **extras)` instead. Output dtype/shape and `additionalArguments`
membership (`ExtraKind.TENSOR` vs `SCALAR`) were read straight off each call site's existing
`outputSizes`/`outputDtypes`/`additionalArguments` construction -- mechanical, not re-derived -- and
ported one module (or gradcheck-script group) at a time, each gated on that module's own
`scripts/gradcheck_*.py` script passing before moving to the next. `getCachedDummyTensor`-built
placeholder tensors already flow through unchanged (Step C's note on `wp_interpolate.py` applies
identically here: nothing about this port touches how those are constructed).

**Correction 2 -- a real gap in Step I's `OperatorSpec`, found by `compSPH/balance.py`.**
`computeCompSPHBalanceTermWarp` launches one thread per *query* particle but writes an output shaped
by the *pair* count (`adjacency.i.shape[0]`), so it passed `numThreads=queryParticles.positions.shape[0]`
explicitly to `warpWrapper2` -- distinct from `outputSizes`. `launchOperator` had no way to express
this: `ThreadSpec` only declared `QUERY_COUNT`, and `_launch` was never even called with a `numThreads`
argument from `operator_spec.py`, so the fallback (`_launch` defaults thread count to the *first
output's* shape when `numThreads` is `None`) would have silently launched this kernel with the wrong
thread count -- pair count instead of query count -- had it been ported as-is. Caught by inspection
before ever running it (same "checked before running" habit as Step F's struct-aliasing hazard), not
by a failing test. Fixed by adding `OperatorSpec.numThreads: Optional[Callable[[ctx, extras], int]] = None`
in `autograd/operator_spec.py` -- `None` (every other spec) preserves today's first-output-shape
fallback exactly; `compSPH/balance.py`'s spec supplies a one-line resolver reading
`ctx.query.positions.shape[0]`. `crk/accel.py`, `crk/dudt.py`, `compSPH/accel.py`, `compSPH/dudt.py`,
and `liu/wp_mat.py` all also pass an explicit `numThreads` to `warpWrapper2` today, but in every one
of those cases the *first* declared output's shape already equals that `numThreads` value (query count,
or -- for `liu/wp_mat.py`, whose "query" struct is actually `referenceParticles` -- the shape of the
`queryPositions` extra, resolved via the same per-output shape-resolver mechanism `OutputSpec` already
had), so the existing fallback covers them correctly without the new field. Only `balance.py` needed it.
This is a **core** change (`warpSPHCore/autograd/operator_spec.py`), landed and verified (core's own
103-test suite, 1 skip, green) before continuing the frontend port.

**Correction 3 -- the `warpWrapper2` shim was NOT deleted.** The plan's closing instruction said to
delete it once the frontend's call sites were migrated; by the time all 24 module sites were ported,
grep found it still has live, legitimate callers outside that scope: `warpSPH/scripts/gradcheck_scalarArg_dt.py`
(a canary proving the `asScalarArg` + `wp.array(dtype=scalar_t)` differentiable-scalar mechanism against
an ad-hoc demo kernel, extensively documented in its own docstring and part of the frontend's
auto-discovered `GRADCHECK_SCRIPTS`) and, in this repo, `scripts/gradcheck_scalar_arg_native.py` (the
original version of that same proof) and `scripts/bench_real_workload.py` (Step H's harness). All
three call `warpWrapper2` directly against one-off demo kernels, not through any of the 24 module
wrappers -- deleting the shim would have broken three working regression/bench scripts to satisfy a
"delete when done" instruction written before this was known. `warpWrapper2` stays as a stable,
tested, lower-level primitive entry point; only the frontend's *production* module wiring moved onto
`launchOperator`. The `defaultStateArguments` tuple path stays for the same reason -- it's what
`warpWrapper2` still builds on.

**Verification (2026-08-18, this machine, CPU -- per [[feedback-3d-cpu-perf]] this project's 2D
gradcheck/physics suites are CPU-appropriate; no 3D workload was run here):**
* This repo's own suite after the `operator_spec.py` change: `pytest tests/` minus the gradcheck
  subprocess file, **103 passed, 1 skipped**.
* Every one of the 13 gradcheck-script groups covering the 24 ported call sites, run standalone
  immediately after that group's port (not just at the end): `gradcheck_crk.py`, `gradcheck_liu.py`,
  `gradcheck_util.py`, `gradcheck_wp_surfaceAware.py`, `gradcheck_dissipation.py`,
  `gradcheck_surfaceDetection.py`, `gradcheck_mdbc.py`, `gradcheck_shockCapturing.py`,
  `gradcheck_deltaSPH.py`, `gradcheck_compSPH.py` (this is the group that exercises the
  `numThreads` fix, via `computeCompSPHBalanceTermWarp`'s 12 sub-cases), `gradcheck_adaptiveSupport.py`,
  `gradcheck_incompressible.py`, `gradcheck_deltaShift.py` -- all green.
* Full frontend re-run at the end, via `pytest`: `tests/test_gradcheck_scripts.py` **15/15 passed**
  (auto-discovered, so this also re-ran the two scripts noted in Correction 3 as still using
  `warpWrapper2` directly, confirming the shim itself is untouched and still correct);
  `tests/test_caseSpec.py` + `tests/test_runner.py` **24/24 passed**; `tests/test_physics.py`
  **51/51 passed** (sod shock tube 1D/2D/3D, Taylor-Green vortex, dambreak, Sedov blast, every
  compressible solver scheme, uniform-lattice density checks).
* `grep -rln "warpWrapper2(" src` in the frontend now returns nothing outside `warpWrapper2`'s own
  definition boundary (it isn't defined there, it's imported from core) -- confirms all 24 production
  call sites moved, none missed.

**What's left:** nothing else scoped to this plan. The frontend's production SPH-operator dispatch
now runs entirely through `SPHContext`/`launchOperator` (directly for the 24 custom-kernel modules,
indirectly through `warpOperation` for the 65 standard-operator call sites); `warpWrapper2` remains
as a deliberately-kept lower-level primitive for demo/bench/canary code, not as frontend production
wiring. A future session wanting to retire `warpWrapper2` entirely would need to port those three
remaining `scripts/` call sites too and accept the docstring/provenance churn that involves -- not
done here because it wasn't asked for and the shim costs nothing to keep.


# 5. Verification

## Correctness

### Tests 4-9 (Step B, Already Passing)
4. **In-place visibility:** `copy_` / `add_` through the torch side is observed by the
   cached view; `data_ptr` stability asserted. ✅
5. **Non-contiguous refusal:** a strided tensor is never cached; a write to it is
   observed on the next acquisition. (Pins the hazard verified in Section 3.1.) ✅
6. **Lifetime / no leak:** with `gc.disable()`, a tensor carrying an attached field is
   collected by refcount alone; the storage is released with it. ✅
7. **Copy/pickle safety:** `copy.deepcopy(tensor)` and a `torch.save`/`load` round trip
   yield a tensor whose next acquisition rebuilds rather than reusing a stale pointer. ✅
8. **`kinds` is mandatory:** constructing a `ParticleState` without `kinds` raises;
   `AllToAll` at `N > 1` no longer reads out of bounds (the previously open item). ✅
9. **Tangent-slot inertness:** with `Field.tangent is None` and one role registered,
   acquisition results are bit-identical to a Field type without those slots -- the
   forward-mode affordances cost nothing and change nothing. ✅

### Tests 1-3 (Step D: Reentrancy)
1. **Reentrancy (the regression that matters).** Call each operator twice on the same
   leaf tensors in one process; compare both gradients against a fresh-process
   single-call run. This is what the deleted cache failed. Write a pytest that calls
   `warpOperation(...)` twice in sequence on the same particle state and verifies
   gradients match a single-call baseline.
2. **Dual-call in-process test:** Repeat any gradcheck script twice without tearing down
   the process; gradients on second call match first-call baseline.
3. **Multi-operator reentrancy:** Call different operators in sequence on same state;
   verify no cross-operator gradient pollution.

### Tests 1-3 (Step E: Full Gradcheck Twice-in-Process)
1. **Full gradcheck suite** via the `gradcheck` skill: Density, Interpolate, Gradient,
   Divergence, Curl, Laplacian, plus `gradcheck_crk_native`, `gradcheck_renorm_native`,
   `gradcheck_covariance_native`, `gradcheck_pinv_native`,
   `gradcheck_scalar_arg_native`. Each **run twice in the same process** from Step E on.
   (Running in subprocess isolation as today is not sufficient; must prove reentrancy.)
2. **Frontend gradchecks:** the 11 `warpSPH/scripts/gradcheck_*.py`, same protocol (twice in process).
3. **Escape hatch bisect:** `WARPSPHCORE_FIELD_CACHE_GRAD=0` forces fresh builds on grad path;
   if any gradcheck fails, run with escape hatch set to isolate cache-specific issues.

### Test 10 (Step C: Null-Field Isolation)
10. **Null-field isolation:** with `WARPSPHCORE_NULL_FILL=sentinel`, the whole operation
   matrix and gradcheck suite still produce finite results -- proving no code path
   reads a disabled correction. (A read from a disabled field shows up as NaN in float output
   or INT_MIN in int output, which fails assertion; a correct path never reads a disabled field.)

### Test 11 (Step F: Forward Equivalence)
11. **Forward equivalence:** `scripts/operation_matrix.py` full sweep (precision x dim x
   jitter x device x traversal x correction) before/after, bit-identical. Bundle assembly
   must produce struct layout bit-identical to prior closure approach.

### Test 12 (Step F: Physics)
12. **Frontend physics:** `warpSPH/tests/test_physics.py`, plus `sod_2d` and a
    Taylor-Green run compared against stored reference output. (Deferred to Step F gate,
    only if frontend test infrastructure is available; optional if frontend has no CI yet.)

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

**Step I (core).** Introduce `OperatorSpec` / `SPHContext` / `launchOperator`. Port all
15 core files that use `warpWrapper2` / `extractStateInfo`. Reimplement `warpWrapper2`
as a thin deprecating shim over `launchOperator` -- it keeps its exact current
signature, so **the frontend keeps working untouched throughout this step**. Gate:
operation matrix bit-identical; gradcheck green through both the new path and the shim;
bench shows no regression and the dtype-probe saving.

**Step J (frontend).** Migrate the 24 `warpWrapper2` sites and the 65 `warpOperation`
sites, module by module, each with the frontend gradcheck and physics suites green.
Convert the sprawling `Optional[...]` parameter lists on wrappers like
`computeCompSPHBalanceTermWarp` into an `SPHContext` plus declared extras while doing
so. Then delete the shim and the old `defaultStateArguments` path from the core.

**Prerequisite for Step J:** give `warpSPH` a `.github/workflows/` that runs its
existing `tests/` -- the suites are there but nothing runs them automatically, and a
65-site migration should not be the first thing to find that out. Skip this step as the
tests are computationally expensive (if they run on CPU only and GPU runners are not available).
Especially the physics ones are expensive, and running them on every push is not necessary, 
especially before this major rework is done. 

Both steps are behind everything else deliberately: the perf work should be landed,
measured and stable before the interface it sits behind starts moving -- and, as of Step H's
insertion, "measured" now explicitly includes real-workload profiling, not just the synthetic
per-call benchmark, so a deeper architectural fix Step H turns up has a chance to land while the
interface is still the old one, rather than being bolted onto the new one under time pressure.
