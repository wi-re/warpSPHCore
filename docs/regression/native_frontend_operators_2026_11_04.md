# Frontend native operator integration, 4 November 2026

The paired frontend branch `codex/native-frontend-operators` reuses native core
CRK preparation, radial kernels, traversal, stable Torch bindings and execution.
Frontend selection remains separate from `WARPSPHCORE_OPERATORS` and defaults
are unchanged. Partial solver routes are explicitly mixed.

Meshless geometry now accepts an optional per-instance `backend`. Its operator
specifications, prepared launcher, closure matvec and multigrid restrictions
follow that selection. Geometry/multigrid/captured-workspace caches include the
backend. Native closure reassignment rejects different CSR index/offset dtypes
instead of narrowing into the older workspace. Native meshless GPU launches
establish the tensor's device before using its current stream. Coarse cell IDs
retain their int32 contract with checked extents/products; CSR edge arithmetic
and query row offsets remain wide.

The existing native canonical-pair flux accepts optional reference gamma and
material fields and a replicated device timestep. It shares the single-gamma
pair implementation while retaining two-sided caloric indices, the audited
Roe/PVRS fallback, cross-material contact faces, mass caps, retry and face guards.
The legacy 17-field binding layout remains accepted; Python supplies 20 fields.
The frontend contributes pointwise conserved/diagnostic primals and routes its
limiter/reconstruction/flux through this implementation. Shared ideal-gas AD
continues through Torch; geometry/rates remain explicitly forward-only.

Qualification includes four native core meshless/closure tests on CPU and CUDA,
60 shared controller contracts on CPU (six existing 3D device exclusions), and
per-instance native/Warp multigrid/cache checks on CPU and CUDA. The frontend
material/reconstruction/1/2/4-chunk operator checks pass on CPU/CUDA, including
FP64 CPU, exact device-timestep graph replay, and rejection of any Warp launch
in the selected native geometry/rates/CFL/primitive route. Existing frontend
accepted-stage, material-contact, conservation, retry, checkpoint and neighbor
cache suites pass. HIP gfx942 compilation of frontend and shared meshless units
passes; no AMD runtime qualification is claimed. No tolerances were relaxed.

The earlier CRK/delta checkpoint is frozen at frontend `94be122` / core
`c6d1ff4`. Three uninstrumented accepted-step repetitions per backend plus
separate exclusive stage runs use Warp 1.19, FP32, input order, Wendland4,
n_h=4, fixed dt=1e-4 and delta sound speed=10. Saved initial/every accepted/final
UID-ordered states pass the maintained comparison. RTX PRO 6000 Blackwell mean
step costs are 20.04→9.76 ms CRKSPH and 7.78→4.89 ms deltaSPH. Four-thread CPU
128² means are 1417.34→567.15 and 516.76→147.60 ms, including slower native
first repetitions. These are mixed executions, not fully native solver claims.

Fixed-list public CRK force/work runs compare identical CSR and 1/2/4-chunk row
orders with explicit policies. The 64³ isentropic GPU fixture has 127,139,840
stored edges, mean degree 485. Native CSR force/store medians are 6.74/7.14 ms;
four chunks cost 7.23/7.64 ms, retaining about 7% overhead against CSR. The
non-isentropic shear fixture fails six of 381 million retained work components
where the entropy split depends on nearly zero estimated-work sign. Force
outputs pass; that retained-vector case remains unqualified at its unchanged
bound. The isentropic run is separate evidence, not a replacement for failure.

Real CUDA allocations test a single live edge above 2^31 and flattened work
element 6,442,450,958 in FP32/FP64, about 60.1 GB peak. This proves address
width, not billion-live-neighbor throughput. Bounded storage/ownership remains
in place and int32/int64 performance tuning is deferred.

Evidence: `$WARPSPH_DEVELOPMENT_ROOT/core/experiments/native-frontend-operators-20261103`.
Immutable phase source/binary snapshots, compiler flags, raw samples, failed
attempts, snapshots and dispatch are retained there. Campaign sources are in
`core/research/native-backend/experiments/frontend-operators` (initial research
commit `459d31f`). The frontend coverage ledger carries reproduction details.
The Isambard run remains stopped; no remote work was launched.

Native owned pair fields now accept `backend='native'`. Byte budgets and dedicated
per-chunk storage remain mandatory; scalar-flattened extents may exceed Warp's
conservative int32 guard. Structured row dispatch revalidates the consuming
backend, rejecting a Warp consumer before output allocation. J capacity remains
bounded by the existing scalar-list planner/collector policy; native full CSR
has its separate wide-addressing route. Twenty owned-field tests pass on CPU and
CUDA. Native CRK requests this storage explicitly; other force families retain
their existing guards. Ownership and communication partitioning are unchanged.

The IISPH conversion exposed an existing FP32 precision sensitivity: a native
Wendland4 product-rule derivative subtracts two O(1) terms to obtain its O(q)
slope. The unchanged dense-Jacobi check failed at 5.44e-5 against 2e-5 while
its Warp control passed. Native now evaluates the identical spline's factored
first, second and third derivatives, including the full RadialJet chain rule.
The original failure remains external evidence. All 16 coupled CPU cases now
pass the original bound. Kernel/independent small-slope checks pass 21 CPU and
21 CUDA cases; seven CUDA reverse-family checks pass. The frontend converted
family suite passes 405 CPU checks (four device skips) and 409 CUDA checks.
The projection/routing suite passes 153 CPU checks, with ten existing skips and
two derivative cases excluded from this primal qualification. No tolerances or
physical equations changed. Timing measurements pinned before this change
retain their original source hashes and are not relabeled as new performance.

PreparedLauncher now recognizes native launch records before converting extras or
allocating Warp outputs. Its borrowed metadata is retained, while Warp tensor
views are created only for an actual Warp consumer. This fixes native output
descriptor handling and removes an unnecessary Warp extent limit from prepared
wide CSR. Two CPU/two CUDA regressions reject every attempted Warp tensor view
in native prepared flat/owned launches. Existing ownership/field/prepared checks
pass 73 CPU (one device skip) and 76 CUDA cases. Both PESPH variants preserve
bitwise flat/chunk/prepared results in eight CPU and eight CUDA checks.


## Complete frontend primal and package checkpoint

Final numerical pins are frontend `148551f` and paired core `1700e1c`. All 100
frontend kernel definitions (65 files, factories included) now have native
primal routes. Late increments cover IISPH, wall moments, surface detection,
Godunov/Gaussian/Inutsuka, material EOS, Morton keys, PESPH, bounded utility and
diagnostic reductions, Liu/mDBC and both ACSPH families. Pair viscosity and
reconstruction share templated physics. Shared core meshless functionality is
reused. Source ownership/manifests, CPU/GPU traversal, bindings and portable
builds follow the existing core structure. Defaults remain Warp; complete
source coverage does not certify every solver/setup/derivative route as fully
native. Unsupported neighborhood AD and method-specific legacy-grid modes
reject explicitly. The frontend coverage ledger specifies these contracts.

Complete source qualification passes 559 CPU cases (five device skips), 564
FP32 CUDA cases and 564 FP64 CUDA cases. Both portable CMake CPU wheels pass
559 cases under strict prebuilt loading. The CMake CUDA frontend wheel paired
with the qualified core GPU development-JIT binary snapshot passes 564 cases,
plus 59 selected installed FP64 independent/dense/width oracles. The GPU core
snapshot is not a separately built core CMake CUDA wheel. HIP gfx942 compiles
all 29 frontend and two shared meshless GPU units; AMD execution is unavailable.
No tolerance was relaxed. Prior baseline failures and discarded probes remain
external evidence, including the native-core uniform-transport and AC no-wall
precision checks documented in the frontend ledger.

Final prepared native calls pass real sparse addressing above 2^31 in FP32 and
FP64 without any Warp tensor view. FP64 peak allocation is 60.13 GB and the
largest flattened work element is 6,442,450,958. Separately, final-source fixed
128³ CUDA lists contain 1,017,118,720 live neighbors (degree 485). Native CSR,
one-, two- and four-chunk force medians are 53.17/53.25/53.59/53.94 ms and work
medians 57.50/57.93/58.18/58.67 ms. Warp force costs 183.60/184.25/185.52/187.21
ms; compatible work is extent-limited for CSR/one chunk and costs 150.69/158.08
ms for two/four chunks. Actual dispatch, complete public calls, seven five-call
repetitions, identical isentropic physical settings and policies are retained.
The same final-source non-isentropic 64³ case still fails six near-zero entropy
sign work components across CSR/1/2/4 chunks at the original bound. It remains
unqualified; the isentropic result does not supersede it.

The earlier large driven 128³ accepted-step checkpoint (frontend `7af0f9c`,
core `cdac0f0`) passes all UID-ordered initial/accepted/final fields at 2e-4;
all 24 controls/profile runs finish at exactly 0.0014999996637925506. Native
CSR/1/2/4-chunk mean costs are 389.61/393.24/391.39/393.64 ms versus Warp
935.56/938.93 ms for two/four chunks (CSR/one-chunk Warp work unsupported).
This trajectory has candidate degree 515, distinct from the degree-485 fixed
list. Source flags/policies are frozen with each checkpoint.

Final-source 128² CPU accepted-state and stage comparisons pass at 2e-4 with
exact final times. One-thread Warp/native medians in ms: CRK 1228.03/637.46,
delta 346.62/338.67, MFM 319.41/330.84, MFV 175.67/197.68. Single-thread MFM
and MFV still regress by 3.6% and 12.5%. Four-thread medians are respectively
1280.07/180.91, 328.97/89.24, 489.07/89.76, 258.44/53.15; raw slower runs and
means remain included. A one-file factored-W4 experiment against the earlier
meshless checkpoint reduced its original 45–58% single-thread regressions,
but did not eliminate them. Index-width tuning remains a separate follow-up.
