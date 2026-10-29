# Whole-row neighbour chunks and multi-GPU execution — core handoff

Canonical paired plan: [warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md](../warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md).
This sibling link assumes the documented paired checkout layout. In another
layout, obtain the canonical file from the frontend repository. Progress has
one indexed row in frontend`PLANS.md`; do not maintain competing checklists here.

Status, 2026-10-29: planned, after a mergeable ACSPH/frontend integration checkpoint.
No chunked backend, distributed simulation or new implementation worktree exists.
Core baseline is`3c56e2f` on`codex/compressible-csr-only` (version0.6.3), based on
shared`main` at`98081c1`. This handoff adds documentation only.

## Integration order

1. Preserve the current frontend ACSPH work until its owner reaches a committed,
   mergeable checkpoint; inspect fresh status in both repositories.
2. Merge this core branch into the actual core integration branch (currently
   `main`), qualify bounded CSR-only/list/stream/offset controls, then merge the
   turbulence frontend branch into frontend`dev` against core>=0.6.3.
3. Record paired integrated SHAs and create separate sibling implementation
   worktrees (suggested branches`codex/chunked-neighbors-multigpu`). The existing
   turbulence checkout remains available for single-GPU physics up to128^3 after
   readiness checks. Do not alter shared installs or source-pinned live jobs.

## Agreed core contract

- Count neighbours for disjoint particle ranges against synchronized reference
  fields; all-gather int32 counts C[N]. Independently scan into int64 P[N+1]
  on every GPU; derive identical whole-row chunks and owners deterministically.
- For owned particles[a,b), local offsets are P[a:b]-P[a], and live edges are
  P[b]-P[a]. Allocate/collect directly into local flat J (global particle IDs),
  plus local W/kappa as needed. No global-edge intermediate or implicit flatten.
- Include stored Verlet acceptance and capacity headroom in budgets; empty rows
  remain owned. Handle oversized rows explicitly. Each local buffer obeys Warp
  extent/stride limits even when the global edge count exceeds INT_MAX.
- Retain local adjacency until coordinated invalidation/rebuild. Keep pair identity
  and CRK work ownership stable during accumulation, or migrate/replay correctly.
  Never assemble the full W tensor on any GPU.
- Particle/reference fields are replicated initially. Exchange results between
  dependent passes, plus global timestep/rebuild/source/failure reductions.
  MFM closure needs distributed iterative vector/reduction handling separately.
- Existing CUDA neighbour lanes and row-chunk ownership are independent. Preserve
  their defaults/ordering while validating chunks, then qualify combinations.

## First deliverables and source anchors

Start with metadata-only partition tests and tiny forced chunks on one device.
Then direct chunk construction/lifecycle, shared operators and native CRK work;
physical two-/four-GPU execution follows single-device parity. Preserve all fresh
stage/integrator/model/forcing choices. Flat APIs remain available; unsupported
large AD/graph/wall paths fail clearly instead of making a monolithic allocation.

Anchors under`src/warpSPHCore/`:
`radiusSearch/csr.py`, `radiusSearch/compactHash/{search,wp_collectNeighbors}.py`,
`radiusSearch/verlet/{build,filter}.py`, `dataTypes/adjacency_t.py`,
`autograd/{arg_extract,prepared,operator_spec,lanes}.py`, `util/stateBundle.py`,
`mfm/{wp_closure,warpBackend}.py` and`ADJACENCY_CAPACITY_PLAN.md` at repository root.
Existing`collection='split'` separates J collection and I fill; it is not the new
whole-row chunk backend. Existing tiled lane kernels are not edge-storage tiles.

Extend count/order/reuse/stream tests in`tests/radiusSearch/`, especially
`test_csrOnly.py`, `test_splitCollection.py`, `test_indexDtypes.py`,
`test_queryOrder.py` and`test_verletValidity.py`. Select int64 offsets before import
with`warpSPHCore_OFFSET_DTYPE=int64`; keep particle IDs/counts int32. The canonical
plan contains all phase acceptance gates and HPC/restart/AD/graph qualifications.
Metadata/aliased high-address tests do not replace real large CUDA storage tests.
