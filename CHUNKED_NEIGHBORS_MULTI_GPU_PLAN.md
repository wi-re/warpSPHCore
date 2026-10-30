# Whole-row neighbour chunks and multi-GPU execution — core handoff

Canonical paired plan: [warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md](../warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md).
This sibling link assumes the documented paired checkout layout. In another
layout, obtain the canonical file from the frontend repository. Progress has
one indexed row in frontend`PLANS.md`; do not maintain competing checklists here.

Status, 2026-10-30: **Phase 4 bounded replicated-state process proof of concept qualified;
local L1 restart and L2 launcher qualified; L3 control/hash slice qualified; payload/workspaces next**, in the paired `chunked-neighbor-planner` workspace,
branch `codex/chunked-neighbor-planner`. Integrated core base is `04e126e`
(version 0.6.3), with Phase1 planner checkpoint `9a1057e`.
`radiusSearch/chunked.py` adds opt-in row-chunk search, resident Verlet reuse,
direct filtering across changed chunk boundaries, generation/configuration
invalidation and logical memory accounting. It reuses the existing numerical
count/collect/filter kernels and adds only an integer global-I fill kernel.
Full core neighbour CPU suites pass 487 checks under int32 offsets and 488 under
int64; 186 distinct CUDA checks pass per offset configuration. Frontend focused
CPU compatibility passes 73 checks. Runtime imports pass. The former static scanner wp_alpha.computeAlpha crash is
now fixed in the paired frontend, with a full clean static pass; see the canonical
plan's resolution note. Older integration results below remain historical.
Commands, expected skips, evidence and limitations are in the canonical plan.
The native compatible-work checkpoint below is now qualified; physical multi-GPU execution remains open.

Transport decision, 2026-10-30: production uses PyTorch `torch.distributed`/c10d
with NCCL, one process per GPU; Gloo CPU tensors provide a multiprocess debugging
path. Shared base NCCL remains 2.29.7; the user-authorized isolated 2.32.3
runtime now passes a two-process/one-GPU communication/Warp-stream smoke with
experimental shared-GPU ranks enabled. The reusable launcher and provenance are
under frontend `.tmp/nccl-shared-gpu-20261030/`. This does not replace physical
multi-GPU gates. See the canonical plan's Phase4 transport section.

## Shared-operator / first solver checkpoint (2026-10-30)

Phase3 now has an explicit host row adapter for shared particle-output operators,
all canonical query/reference correction fields, and chunked prepared launches.
Existing floating kernels and the Warp struct ABI remain unchanged. Extra tensors
declare query/reference/replicated identity, including enabled optional fields.
In-place reuse invalidates old storage leases; independent old buffers remain
valid. AD/capture, partial-owner outputs and unported specs reject explicitly.

Frontend support/dissipation integration and five tiny two-step eager Monaghan
fixtures qualify stored-field, fresh-stage and source-budget parity. Select an
explicit chunked Verlet list through a Python case build hook; defaults and CLI
selection remain unchanged. The native W checkpoint followed this shared-operator work; checkpoint/restart
and physical distributed execution remain open. Per-side pressure and wall closure remain gated. The canonical plan has
the full CPU/CUDA/int32/int64/FP64 test matrix and evidence directory.

## Native compatible-work checkpoint (2026-10-30)

`ChunkedPairField` pins dedicated pair buffers to one adjacency generation/lease.
The host operator ABI maps owned chunk extras and global query-row bases, with
explicit bounded unread scratch outputs. Frontend native eager CRKSPH/CompSPH
force -> half velocity -> owned W -> ordered final reduction now passes strict
work/stage/field parity and forced changed-generation/layout replay on CPU/CUDA,
including both local offset widths, CPU FP64 and fixed 32-lane CUDA controls.
The test matrix also covers prepared fields, storage aliases/budgets, empty and
padded storage, unsupported contexts and delayed CUDA producers. The discovered
flat CRK-term/work-reduction stream dependency is repaired without changing
floating kernels. Raw failure and successful rerun evidence remain available.

No global J/W compatibility view or default/CLI change was added. Single-device
total resident work is unchanged apart from padding; kernel output scratch is
bounded while particle outputs remain O(N). See the canonical plan for measured
storage/timing, exact qualifications and limitations. Native special controls,
AD/graphs/walls/materials, checkpoint/restart and partial-owner output exchange
remain gated. The NCCL/Gloo process checkpoint below is qualified locally;
physical multi-GPU and large-edge/capacity gates remain open.

## Replicated-state process checkpoint (2026-10-30)

`warpSPHCore.distributed.RowChunkProcessGroup` explicitly binds an existing c10d
NCCL/CUDA or Gloo/CPU group. Distributed search counts only the local counting
range, assembles replicated int32 counts/int64 prefix, validates configuration/
generation/grid/partition agreement, then allocates/collects only owned chunks.
Resident validity/rebuild is coordinated. Row dispatch and prepared operators
exchange disjoint particle outputs before dependent reads, checking reference
inputs and operation/layout contracts. Pair fields stay owned; distributed
filtering/migration is still gated. There is no implicit/default process group.

Separate CPU processes (two/three ranks), empty/uneven owners and deliberate
contracts/fatal-peer tests pass. Two real NCCL ranks share one GPU through the
isolated 2.32.3 runtime; bounded eager frontend Monaghan/native CRKSPH/CompSPH
solver stages, owned work and forced replay match serial references. The longer
16-step native tests cross OU knots and retain exact simulated time/history/
particle/work parity after fixing a tensor-dt-to-Python precision loss in the
frontend adapter. Timings are diagnostic under ACSPH contention. The canonical
plan records the exact matrix, observed memory, reproduction/failure evidence,
API/dependency schedule and physical-device/site/capacity limitations.

## Accepted-step native restart checkpoint (2026-10-30)

`radiusSearch/chunkedState.py` exports/restores a checksummed versioned search
recipe: frozen query/reference snapshots, domain, counts, generation and ordering
metadata. Aliases and deterministic row traversal survive changed chunk budgets,
owner counts and int32-to-int64 local offset configuration. Restore recounts and
agrees on the complete target plan before allocating only owned J/I. No pair
array, prefix, communicator or device ordinal is serialized. Precision and
traversal-version mismatches reject explicitly.

The paired frontend adds explicit accepted-step checkpoint APIs, atomic rank-zero
HDF5 publication and a runner resume branch for eager driven Monaghan/CRKSPH/
CompSPH. CPU Gloo and shared-GPU NCCL controls compare fresh stages, fields,
per-row J/W hashes and full forcing history exactly (existing source-budget
reduction tolerance retained). Empty owners and recoverable IO failures pass.
The canonical plan records the full matrix, startup seed/restored-storage overlap,
source-pinned evidence and API. Legacy flat import and physical-device restart
remain open; all seven user-approved local steps are ordered in that plan.

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
Next: distributed count/collect, process groups and particle-field exchanges
between dependent passes while J/W remain owned. Physical two-/four-GPU
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

## Local continuation update (2026-10-30)

The recipe encoder now also accepts explicit frozen inputs/counts without pair
arrays. The paired frontend imports legacy version-1 CSR/COO history, validates
int64 scans and verifies each owned J/I slice before installation. Bounded CPU
and shared-GPU NCCL controls preserve exact continuation under changed policies,
ranks and local offsets. Legacy missing search-policy/checksum metadata and
other limitations remain explicit in the canonical plan.

The frontend local launcher/output adapter is qualified for the three driven
native schemes; rank-zero accepted checkpoints/diagnostics, source/runtime/rank
manifests, recoverable IO/capacity failures and a locally checked Slurm template
are available. MPI/ROCm builds are explicitly deferred. L3 control/hash overhead
is next; distributed filtering, broader compatibility, actual wide-edge access
and physical multi-GPU/site/capacity/performance gates remain open.

## L3 control/validation checkpoint (2026-10-30)

Explicit exact/contract replica checks now separate per-operator content hashing
from layout contracts, while search/rebuild and frontend accepted/checkpoint
boundaries force exact data agreement. Exact remains default; fast contract
mode assumes the dependency schedule maintains in-stage replica values. Control
packets and pinned host staging are reused; local validation/JSON contracts,
scalar clone validation and complete operator output batches need fewer
handshakes. Data broadcasts and floating kernels remain unchanged.

Matched frozen7c09724 controls6756→3288 per rank; FP32 hashed bytes7,860,720
→222,528 with contract mode;20,268 explicit control Tensor factories→0 after
construction. CPU FP32/FP64 and shared-GPU NCCL exact stages/fields/W/time/OU
controls pass, including nondefault/alternating streams, recoverable/fatal
failures, long forced replay and changed-mode restart. See the canonical plan
for precise scope, limitations and source-pinned evidence. Next L3 work is
compatible particle payload packing and bounded workspace reuse.
