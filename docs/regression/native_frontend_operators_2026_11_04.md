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
