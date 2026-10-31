# Whole-row neighbour chunks and multi-GPU execution — core handoff

Canonical paired plan: [warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md](../warpSPH/CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md).
This sibling link assumes the documented paired checkout layout. In another
layout, obtain the canonical file from the frontend repository. Progress has
one indexed row in frontend`PLANS.md`; do not maintain competing checklists here.

Status, 2026-10-31: **Bounded local L1–L7, allocation recovery and launch geometry
qualified; remaining scheme compatibility active. Native PESPH driving/checkpoints
are qualified in the frontend. C2 now includes row-owned MFM/MFV geometry,
projected CG/MG closure, gradients, canonical fluxes and native continuation.**
Long-flow/physical multi-GPU/site/performance gates remain held. Active paired
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
are available. MPI/ROCm builds are explicitly deferred. L3 control/hash and packed-payload work
is qualified below; distributed filtering is next. Broader compatibility, actual wide-edge access
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
for precise scope, limitations and source-pinned evidence. The following checkpoint qualifies
compatible particle payload packing and bounded workspace reuse.

## L3 particle packing/workspace checkpoint (2026-10-30)

Compatible same-dtype particle outputs now share bounded whole-row packets, using
one lazy fixed-capacity workspace per group (default 4 MiB, constructor option
`particleWorkspaceBytes`; zero disables packing). Singleton/oversized rows and
packing that would increase broadcast calls use the direct path. Outputs retain
their own storage, with no pair migration or arithmetic/dtype conversion. Complete
layouts, alias topology, capacity and allocation agree before any payload;
partial/retyped overlap and mismatched aliases/budgets reject coherently.

A CUDA event fences the last unpack before scratch reuse on another stream;
`record_stream` registers allocator lifetime, without an extra host wait. Mixed
shape/type and NaN/signed-zero bit tests, tiled/empty owners, allocation failure,
retained output and delayed stream reuse pass. The cache has one allocation and
no per-shape/tag entries. Existing exact/contract semantics remain unchanged.

Matched frozen 3742f10 → packed payload broadcasts 1424→968 per rank (32.0% fewer),
with 3288 controls and unchanged logical payload bytes. CPU FP32/FP64 and shared-GPU
NCCL qualifiers preserve exact fresh stages/owned W/fields/clock/forcing, including
int64 offsets, 32 lanes, 16-step replay and changed-rank/mode restart. Full frontend
runtime/static import checks pass. Timings remain diagnostic under ACSPH contention.
The canonical plan records the full matrix, source-pinned evidence and memory/API
limits. L3 is locally qualified; L4 distributed filtering/redistribution is next.
Physical multi-GPU/site/capacity/performance and L5–L7 remain open; MPI/ROCm builds
remain deferred. No shared sources or installed stack were changed.

## L4 distributed filtering/redistribution checkpoint (2026-10-30)

Resident filtering now counts only current owner rows, agrees the filtered plan
and streams one bounded filtered source-chunk J interval at a time directly into
new target chunks through explicit c10d send/recv/local copy. Whole-row traversal
and optional global I are preserved, without a global pair gather or W migration.
All ranks validate allocation/transfer contracts before payload. Old leases/J/W
remain valid; changed list identity retains native fresh-stage replay semantics.
Peak logical scratch is one source chunk's filtered J/local offsets, in addition
to overlapping old/new owned lists and O(N) particle/row metadata.

CPU2/3 ranks, exact/contract, all support modes,2D/3D, empty owners, split/merged
chunks, stored-I changes, six recoverable failures and source/work retention pass.
CPU FP64/int64 and shared-GPU NCCL FP32 with both offsets preserve exact native
CRKSPH/CompSPH filtering/replay (8 fresh RHS per accepted step). Final core/search
suite191 pass,187 expected skips;4 frontend replay drivers pass. Full frontend
import scanner passes. The canonical plan/evidence records exact counts, timings,
source variants and limits. L4 is locally qualified; L5 stress is active and L6
compatibility follows, as authorized by the user. Physical devices/site/scaling,
MPI/ROCm builds and large-memory L7 remain deferred.

## L5 bounded stability/failure checkpoint (2026-10-30)

A durable seeded worker exercises4/5 CPU ranks, repeated population/geometry/
capacity/alignment/policy/owner changes, empty owners, in-place leases, exact flat
filter parity, retained W/J and obsolete pair Tensor lifetime. Transfer-contract
failures recover; fatal sender/receiver tests require launcher peer termination.
NCCL2-rank exact/contract48-cycle controls keep one4 KiB packet cache and no old
pair Tensors; CPU FP64/int64 repeats48 cycles on5 ranks. Torch allocator spans are
bounded locally, without a backend/process peak or long-duration claim.

The frontend now coordinates native work-input allocation before payload, with
one-rank RHS OOM/retry controls and unchanged numerical Warp ASTs. Both native
schemes pass64 filtering/replay steps /512 fresh RHS per scheme/rank across54 OU
knots, exact state/work/clock/history and zero steady allocated-memory span. Native
checkpoint recovery1→4→5 ranks/int64/empty owners and injected IO controls pass.
Full imports pass. The canonical plan has precise matrices, wall/memory limits,
source-pinned evidence and paired heads. L5 is bounded-locally qualified; L6
compatibility follows. Physical devices/site/scaling, MPI/ROCm and L7 stay open.

## L6 bounded compatibility checkpoint (2026-10-30)

The frontend now qualifies eager GSPH and InutsukaGSPH through explicit row roles,
integer global self-pair identity, Gaussian density and all3 prepared Godunov
operators. The shared startup/launcher/native-checkpoint gate admits5 schemes;
legacy CSR/COO stays scoped to the original3. Floating Warp ASTs remain equal to
L5 after integer rowStart normalization, and all16 flat Godunov gradchecks pass.

Core rejects forward AD before detached search snapshots, rejects actual CUDA
capture before any control callback/collective, and agrees effective kernel/lane
selection before owned compute. Existing tiled kernels and scalar Godunov fallback
pass global32-lane controls; different effective settings reject before launch.
Final matrix:12 rank reports,24 operator configurations each,15 CPU/26 CUDA
rejected contexts. Five-scheme final-source NCCL launcher/native IO repeats15
serial jobs with exact field/time/OU parity across2→1 ranks and changed row caps.
The canonical plan records292 formal passes/245 expected skips, full process/
solver/offset/FP64 matrices, exact source variants and ignored evidence.

L4–L6 meet their bounded local exits. The maintained support matrix is
`../warpSPH/docs/chunked_distributed_support.md`. MFM/MFV/PESPH, walls/materials,
chunked AD/graphs, physical GPUs/site/scaling and large-edge capacity gates remain
open. L7 and MPI/ROCm builds remain deferred; no shared installs or ACSPH sources
were changed, and contended single-GPU timings remain diagnostic.

## L7 real wide-edge debug access (2026-10-30)

The user released the GPU and authorized large single-node/shared-GPU debug
runs. After checking92,094 MiB free, a new durable opt-in worker physically
collects2,209,000,000 neighbours for47,000 coincident particles into33 separate
J chunks. It reads every J, stamps/verifies actual3-component owned work through
prepared row dispatch and runs production density eager/prepared against an
independent O(N) one-row flat oracle. Device probes verify global indices
2^31−1,2^31 and the final edge; the global prefix is int64, local offsets are
qualified separately at int32 and int64. No production library change was needed.

Standalone and two shared-GPU NCCL2.32.3 cases pass. Filtering creates66 target
chunks, preserves all old J/work and transfers137,240,000 bytes across changed
owner boundaries. Standalone Torch allocated peak41.182 GiB; per-rank peaks in
the NCCL cases are21.380/20.307 GiB. These are allocator/debug measurements.
The complete core process suite passes14 drivers, including3 new small CPU/Gloo
oracle cases. Full frontend imports pass. Commands, raw evidence, source pins
and final paired heads are in frontend `.tmp/l7-wide-20261030/`.

L1–L7 local exits are now qualified; consult the canonical plan/support guide.
Realistic spatial/developed-flow capacity, wide-edge solver trajectories, large
individual chunks, physical GPU/site/scaling, broader compatibility and MPI/ROCm
remain separate gates. This capacity/addressing fixture has no flow integration.

An additional50,000-particle fixture passes2,500,000,000 actual J/work entries
across38 chunks, with final global chunk base2,482,700,000 above INT_MAX. Both
offset widths and standalone/two NCCL ranks pass eager/prepared density, full
audits and filtering. Standalone allocated peak46.578 GiB. In total8 large
cohorts /12 rank reports qualify both crossing inside a chunk and whole chunks
starting above the limit; full spatial/flow capacity remains open.

## Overnight real3D CRKSPH flow (2026-10-31; running)

User authorizes32³/64³/96³/128³ and a measured >INT_MAX flow to t4, sparse
checkpoints, and a conditional96³/128³ continuation to t40 if measured runtimes
fit10–12+hours. A detached serial supervisor launched00:01UTC31Oct with a
12.5hour budget and frozen copies of frontend/core/integrators/plotting. Native
fresh midpoint/B7/nh4/OU/bath choices match the established pilot. The single
GPU runs one process with owned chunks/int64 offsets and normal lane defaults.
Small3D flat/chunked/restart/endpoint/video gates pass.32³ has completed t4;64³
is active. Initial515 stored neighbours per particle imply164³ is the likely
wide target; actual count/memory and full RHS gates will confirm it. No library
numerical sources were changed. The canonical frontend plan and live evidence
`.tmp/crksph-3d-overnight-20261030/status.json` record the queue/continuation rules.
The queue is ongoing; larger flow capacity and statistical qualification are
not yet established. Runtime snapshots are isolated from subsequent edits.

## Overnight outcome (2026-10-31)

32³/64³/96³/128³ reach t4 with clean RHS/energy/mass gates, in39s/8min/41min/
2h18min loop time. Their native checkpoints pass independent finite/positive/
endpoint/continuation-dt audits and all941 frozen hashes remain unchanged.
164³ builds2,271,636,160 neighbours and executes245 full wide RHS evaluations,
then hits the campaign work-allocation headroom guard at t0.1038 after122
accepted steps. Peak allocated60.014 GiB; no invalid RHS was observed. The
process also exitsSIGSEGV after the recorded Python guard error. There is no
wide endpoint/checkpoint; the queue stopped03:17UTC and did not launch t40.
Next investigation: work-buffer lifetime, guard free/cache accounting and safe
cleanup, without assuming actual OOM/overflow/leak from the current trace.
Canonical frontend plan and ignored campaign outcome.json contain exact results.

## Accepted-state allocation recovery (2026-10-31)

Failure-only J/pair-field receipts report requested batch capacity, completed
chunks and CUDA live/reserved/free memory; partial new buffers are released.
Distributed local-phase allocation errors use the existing512byte control
packet and reach every live peer as DistributedAllocationError (also a
DistributedContractError/MemoryError). Other contract faults retain their type.
Core search/recipe/planner/pair-field CPU regressions270pass/212skip; focused
failure/control tests6pass/2skip including two/three Gloo ranks in both modes.

Frontend recovery now retains and atomically saves the accepted particle state
after an eager chunked integrator allocation failure, with forcing rollback
and safe progress/renderer/output cleanup. The former164³ headroom failure
is reproduced as raw-free-only guard accounting:42.08GB cached was ignored;
projected live53.53GB was below the78GiB cap. No allocator OOM/leak was seen.
The oldSIGSEGV reproduces in tqdm.__del__ at shutdown; explicit progress close
fixes the injected CUDA/video failure. A new bounded164³ run passes the old
rejection, saves on injected failure after124accepted steps, and resumes to130;
29 native datasets match the earlier-checkpoint control exactly. No numerical
method/stage-reuse change. Full large t4 and96³→t40 are held for overnight
while development work remains. Canonical frontend plan/evidence under
`.tmp/allocation-recovery-20261031/` track the remaining GPU/profile gates.

## Scalar work launch occupancy (2026-10-31)

Matched single-process64³ profiling isolates the CRK work slowdown to block
geometry:149registers/thread at256threads admits one block/SM. A one-chunk
control matches flat work time; smaller chunks waste the last scheduling wave.
Explicit128threads admits three blocks/SM and reduces work51.47→30.47ms/step
without changing arithmetic or row traversal. Steady total185.08→163.90ms/step
on this early-flow fixture; full late-flow/site performance is unqualified.

Core OperatorSpec adds optional validated blockDim, forwarded through ordinary,
prepared and autograd launch paths. None retains historical256 scalar threads;
cooperating tiled lanes and CPU behavior stay fixed. Distributed contracts
include the effective kernel/lane/block selection and reject differing CUDA
blocks before payload launches. Only frontend CRK work selects128. Prototype
module-option-only trials were ineffective and remain excluded from conclusions.

All saved64³ tensors/source budgets match the old256-thread endpoint bitwise.
Production wide failure/restart to step132 at164³ matches29 native datasets
exactly against the old launch geometry. CPU declared/prepared/block/reverse
checks pass; final CUDA FP64/NCCL admission is tracked in the canonical front
plan and `.tmp/allocation-recovery-20261031/`. Long flow gates remain overnight.

Final local admission passes: CUDA block ABI17tests including prepared/reverse,
FP32/FP64 native compatible work/replay8tests each and nondefault-stream/32lane
controls2each. Two shared-GPU NCCL ranks pass core operators/errors including
block mismatch; native failure/checkpoint/restart passes with128-thread work.
Scanner:568runtime modules/6105imports/1317files. Wide production recovery
step132 remains exact against256-thread control. No full-duration large run
is launched; those are held for overnight. Physical multi-GPU and other site
backends remain separate gates.
