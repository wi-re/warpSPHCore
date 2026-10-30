# Whole-row neighbour chunks and multi-GPU execution — core handoff

Canonical paired plan: [warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md](../warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md).
This sibling link assumes the documented paired checkout layout. In another
layout, obtain the canonical file from the frontend repository. Progress has
one indexed row in frontend`PLANS.md`; do not maintain competing checklists here.

Status, 2026-10-30: **Phase 2 standalone single-device construction/lifecycle
qualified on CPU and CUDA**, in the paired `chunked-neighbor-planner` workspace,
branch `codex/chunked-neighbor-planner`. Integrated core base is `04e126e`
(version 0.6.3), with Phase1 planner checkpoint `9a1057e`.
`radiusSearch/chunked.py` adds opt-in row-chunk search, resident Verlet reuse,
direct filtering across changed chunk boundaries, generation/configuration
invalidation and logical memory accounting. It reuses the existing numerical
count/collect/filter kernels and adds only an integer global-I fill kernel.
Full core neighbour CPU suites pass 487 checks under int32 offsets and 488 under
int64; 186 distinct CUDA checks pass per offset configuration. Frontend focused
CPU compatibility passes 73 checks. Runtime imports pass; the static scanner's
pre-existing wp_alpha.computeAlpha crash also reproduces against base 04e126e.
Commands, expected skips, evidence and limitations are in the canonical plan.
Native operators/compatible work and physical multi-GPU execution remain open.

Transport decision, 2026-10-30: production uses PyTorch `torch.distributed`/c10d
with NCCL, one process per GPU; Gloo CPU tensors provide a multiprocess debugging
path. Shared base NCCL remains 2.29.7; the user-authorized isolated 2.32.3
runtime now passes a two-process/one-GPU communication/Warp-stream smoke with
experimental shared-GPU ranks enabled. The reusable launcher and provenance are
under frontend `.tmp/nccl-shared-gpu-20261030/`. This does not replace physical
multi-GPU gates. See the canonical plan's Phase4 transport section.

## Integration order (completed history)

1. Preserve the current frontend ACSPH work until its owner reaches a committed,
   mergeable checkpoint; inspect fresh status in both repositories.
2. Merge this core branch into the actual core integration branch (currently
   `main`), qualify bounded CSR-only/list/stream/offset controls, then merge the
   turbulence frontend branch into frontend`dev` against core>=0.6.3.
3. Record paired integrated SHAs and create one paired implementation workspace:
   one frontend checkout and one sibling core checkout (suggested branches
   `codex/chunked-neighbors-multigpu`). Row chunks first, then multi-GPU execution
   proceed sequentially in these same checkouts, not in separate phase worktrees. The existing
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

Metadata planning and direct single-device construction/lifecycle are qualified.
Next: shared support/density/correction/gradient operators and native compatible
work, with explicit mapping of every query/reference field. Physical two-/four-GPU
execution follows solver parity. Preserve all fresh
stage/integrator/model/forcing choices. Flat APIs remain available; unsupported
large AD/graph/wall paths fail clearly instead of making a monolithic allocation.

Anchors under`src/warpSPHCore/`:
`radiusSearch/{rowChunks,chunked,csr}.py`,
`radiusSearch/compactHash/{search,wp_collectNeighbors}.py`,
`radiusSearch/verlet/{build,filter}.py`, `dataTypes/adjacency_t.py`,
`autograd/{arg_extract,prepared,operator_spec,lanes}.py`, `util/stateBundle.py`,
`mfm/{wp_closure,warpBackend}.py` and`ADJACENCY_CAPACITY_PLAN.md` at repository root.
Existing`collection='split'` separates J collection and I fill; it is not the new
whole-row chunk backend. Existing tiled lane kernels are not edge-storage tiles.

Chunk-specific suites: `tests/radiusSearch/test_rowChunks.py` and
`tests/radiusSearch/test_chunkedSearch.py`. Existing compatibility suites include
`test_csrOnly.py`, `test_splitCollection.py`, `test_indexDtypes.py`,
`test_queryOrder.py` and`test_verletValidity.py`. Select int64 offsets before import
with`warpSPHCore_OFFSET_DTYPE=int64`; keep particle IDs/counts int32. The canonical
plan contains all phase acceptance gates and HPC/restart/AD/graph qualifications.
Metadata/aliased high-address tests do not replace real large CUDA storage tests.
