# Adjacency capacity padding (warpSPHCore) — plan / note

Status (2026-10-13): **the contract is in** (user: "no time like the present"): `i`/`j` renamed
`paddedI`/`paddedJ` (every old use fails loudly), `numEdges` host int, `livePairs()` exact views,
`CapacityPolicy` (10 % headroom, keep while within [50 %, 100 %] of the prior capacity), padding on by
default in `radiusSearchCompactHashMap` / `buildVerletList` / `filterVerletList` (`capacity=None` = exact),
all consumers in both repos moved, tests in `tests/radiusSearch/test_adjacencyCapacity.py`.
**Not done (phase 5 of ACSPH_REFACTOR_PLAN): in-place buffer reuse and the sync-free rebuild**, see
"Graph contract" below. Rebuilds still allocate fresh buffers (so the address changes) and still read the
edge total on the host once.

## Why

Every neighbour-list build and every Verlet filter sizes its output exactly, which needs the
total edge count on the host:

| Site | Sync | When |
|---|---|---|
| `radiusSearch/compactHash/search.py:90` `torch.sum(edge_count_t).cpu().item()` | allocate | every rebuild |
| `radiusSearch/verlet/filter.py:40` `torch.sum(edge_count_t).cpu().item()` | allocate | every `filterVerletList` call (per step) |
| `radiusSearch/verlet/build.py:104` `shouldRebuild_t.item()` | validity check | every step (deferrable via `deferVerletChecks`) |

Fresh exact-size `i` / `j` tensors also change shape from step to step, which is what stops a
CUDA graph (or a persistent `capture_while` loop) from spanning a Verlet generation, and it
reallocates on every rebuild. Fixed-capacity buffers remove the allocate syncs and make the
shapes static.

## Design (user decisions)

1. **Capacity**: allocate `i`, `j` with ~10 % headroom over the real edge count. A rebuild or
   filter writes in place while the new count fits. Reallocate only
   * on overflow (the count kernel compares against capacity and sets a device flag; the flag is
     read at the Verlet validity check, which already syncs — so no new sync), or
   * when the real count falls below ~50 % of capacity (shrink), same check.
   Numbers (10 %, 50 %) are parameters, not constants.
2. **Padding convention: `i = j = -1`, no dummy particle.** A dead particle with no physics would
   be internal state leaking into exported traces. The tail beyond the real count holds `-1`.
3. **Per-particle CSR is untouched**: `edgeOffsets` / `numNeighbors` never address the tail, so
   every CSR kernel is unaffected. Only raw flat `i`/`j` consumers see the padding.
4. **`numel()` is never a count.** `adjacency.i` is already the velocity-Verlet superset (not the
   active pairs); with padding it is also longer than the superset. Add an explicit
   `numEdges` (device scalar, optionally host-cached when known) to `AdjacencyList` if any
   consumer needs the real superset size; do not derive it from the tensor length.
5. **Raw `i`/`j` interactions stay a development convenience.** New code should use the CSR
   kernels; the dense `i` tensor is a removal candidate once the remaining raw users are
   migrated. Padding must not push anyone to build more on it.
6. Inlets/outlets (changing particle count) are out of scope.

## Graph contract: two invalidation cases (user, 2026-10-13)

A rebuild (outside any graph, between replays) ends in exactly one of:

1. **Same padded size**: the new list is written *into the same buffers* (same shape and same device
   addresses), so the captured graph stays valid and is simply replayed.
2. **New padded size** (the count overflowed the capacity, or fell below `shrinkBelow`): new, larger
   (or smaller) buffers; the graph is re-captured once and keeps working after. Growth is geometric
   (headroom), shrink has hysteresis, so this is rare.

What this needs beyond what is in now:
* **In-place reuse** (`CapacityPolicy.inPlace`, off by default): a graph bakes in pointers, so "same shape"
  alone is not enough. Today every rebuild allocates, which is case 2 every time.
* In-place mutation breaks anything that keys on identity or holds an old list: the fluid-pair cache in
  `cases/common/weaklyCompressible.py` (`cached[0] is adjacency.paddedI`) and compSPH's retained per-stage
  adjacencies (COMPRESSIBLE_SPH_PLAN X6 removes the latter). Audit holders before switching it on.
* Kernels inside a graph take the *padded* length and guard `if edgeI[tid] < 0: return`; they must not
  slice with `livePairs()` or read `numEdges` (host ints change per rebuild, graphs freeze them). `livePairs()`
  is for eager consumers only.
* The fits-or-not decision stays on the host at the rebuild (one sync per rebuild, outside the graph); the
  per-step sync-free path is the validity flag, already deferrable (`deferVerletChecks`).

## Consumer rule (independent of the core change; can land first)

A raw-pair consumer must (a) test `0 <= i < n` (and `j`) before using a pair — a no-op on today's
adjacency, so it is safe to add now — and (b) still test actual interaction (`dist < h`) or
Verlet membership, since the list is a superset. In torch code, **mask before gathering**: a
gather with index `-1` silently wraps to the last element, it does not raise.

Known raw `adjacency.i/.j` users to audit and guard (grep `adjacency\.[ij]\b`, 28 hits at time of
writing):

* warpSPH: `modules/artificialCompressible/wallClosure.py` (`wallBounceSetup` — to be retired as a
  CSR wall-neighbour kernel, not guarded), `modules/crk/accel.py`, `modules/compSPH/{accel,balance}.py`,
  `modules/compressibleWall/wall.py`, `modules/analyticBoundary/{wallPressureMLS,wallViscosity}.py`,
  `modules/shifting/{fickian,implicitShifting,wp_implicitShifting,wrapper}.py`,
  `modules/shockCapturing/{Wadsley2017,wp_readHayfield}.py`, `modules/incompressible/compactProjection.py`,
  `modules/surfaceDetection/colorFieldDetection.py`, `modules/reconstruction/diagnostics.py`,
  `math/scatter.py`, `schemes/compSPH.py`, `cases/common/weaklyCompressible.py`, `io/hdf5.py`
* warpSPHCore: `mfm/{wp_mfm,wp_closure,warpBackend}.py`, `polyfit/polyfit.py`,
  `coreOperations/{wp_densityHVP,_jvpCommon}.py`, `autograd/arg_extract.py`,
  `radiusSearch/{verlet/wp_updateVerlet,small/wp_radius_small}.py`

Warp kernels that loop `tid < i.shape[0]`: add `if i[tid] < 0: return`. `io/hdf5.py` must
write only the real edges (or none; adjacency is not part of a particle trace).

## Steps

1. (Anytime, warpSPH side) add the `0 <= i < n` guards to the raw users above; bitwise no-op.
2. Count kernels take `capacity` and set an overflow flag instead of requiring the host total;
   collect / update-Verlet kernels fill `-1` into the tail.
3. `AdjacencyList` gains `capacity`-aware fields (`numEdges`, `overflow` flag) — all optional,
   default `None` meaning "exact size, as today".
4. `compactHash` search and `filterVerletList` take `capacity=None | CapacityPolicy(headroom=0.1,
   shrinkBelow=0.5)`; default `None` keeps today's behaviour and tests.
5. The overflow / shrink decision is folded into the existing validity read (`build.py:104`) and
   `deferVerletChecks` flags; on overflow the step re-runs eagerly with a larger buffer (same
   mechanism as the deferred rebuild).
6. Tests: padded vs exact list give bitwise identical results through every CSR kernel; a raw-pair
   consumer given a padded list matches the exact one; overflow and shrink paths; no host sync
   in the filter / rebuild with padding on (call audit).
7. Then the warpSPH phase-5 use: persistent buffers, graph / `capture_while` across a Verlet
   generation.

## Risks

* A raw consumer that misses the guard reads index `-1` (wraps silently in torch, out-of-bounds in
  warp). Mitigation: step 1 first, plus a debug mode that fills the tail with `-1` even when
  nothing needs it, so tests exercise the guards.
* Headroom too small → frequent overflow re-runs; tune from the call audit on dam break / sloshing.
* Shrinking policy must not thrash around the 50 % line (hysteresis: shrink to ~65 % fill).
