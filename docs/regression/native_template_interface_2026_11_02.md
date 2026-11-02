# Initial native neighborhood template interface — 2026-11-02

The initial implementation works across existing operator shapes and neighborhood
representations. The matched direct-loop control shows no systematic template
penalty. This is sufficient to start adopting the contract, while retaining
operation-specific scheduling and loaders. It is not evidence that one generic
GPU schedule should replace every optimized kernel.

Implementation worktree: `/home/lu26029/.codex/worktrees/native-cluster-pairs/warpSPHCore`,
branch `codex/native-cluster-pairs`, based on `b0f9dad44f90663cbd0dc07c30c61aae1497cd42`.
Production operator registration/dispatch is unchanged; the private experiment is
in `experiments/template_interface`. No frontend or concurrent-branch changes
are included.

## Interface and application

Shared headers live in `native/csrc/execution`: `neighborhood.h`,
`neighborhood.cuh`, `neighborhoodAdjoint.h` and `sphFields.h`. Function policies
live in their owning module's `native/csrc_<function>` directory. The
[experiment README](../../experiments/template_interface/README.md) documents the
contract and reproduction commands.

| Policy | Accumulator | Finalization/output |
|---|---|---|
| Density | 1 sum | Store density |
| Scalar/vector interpolation | F sums | Store field |
| Scalar Difference gradient / vector Jacobian | F×D sums | Field-first gradient layout |
| Covariance | D² sums + count | Store matrix; count optional |
| Normalized interpolation | F+1 sums | Divide numerator by denominator; optionally save both |
| CRK moments | 1+2D+2D²+D³ sums + count | Store all six moment tensors; count optional |
| Maximum selector example | value + winning index | Store selected value/index; lowest storage index wins ties |

The generic row prepares a query, initializes the operation's accumulator,
traverses references through a loader, reduces the declared accumulator members,
and calls `finish`. CSR32, CSR64, contiguous source ranges and shared-memory
cell tiles execute the same policies. Shared-cell execution owns its complete
source sweep and stages records; it is a distinct executor rather than a
scalar-ID iterator forced onto tiling.

Null pointers express absent fields, saved states, output gradients, tangents
and diagnostics. There are no CSR scans, device-to-host validation copies or
debug switches in the execution path. Planning and verification are separate.

## AD and implicit-solver capability

Manual first derivatives are qualified for Wendland2/Gather density, scalar and
vector interpolation, Difference gradient/Jacobian and scalar normalized
interpolation. JVP uses CSR; VJP supports query-owned CSR plus reference-owned
transposed CSR and a matched atomic-scatter implementation. The owned passes
preserve the original directed pair and support nonsymmetric neighborhoods.
Field-only requests select a narrower specialization and skip unused passes.

Nonlinear finalization has a separate pullback: normalized interpolation first
converts output cotangents into numerator/denominator cotangents once per query,
then the pair pullback consumes those values. This prevents doing finalizer
divisions once per edge. New policies still require manual mathematical
JVP/VJP definitions; the template is not an automatic C++ differentiator.

`implicit_demo.py` uses the native field-only JVP in Newton-GMRES and VJP in an
implicit transpose solve, without recording the nonlinear/Krylov iterations.
CPU and CUDA converge in three Newton iterations, with residual ≈2.84e-13.
The implicit parameter derivative agrees with an independent dense solve to
8.9e-16; Jv and Jᵀv agree to below 5e-16 in norm. Dense matrices are formed only
for verification after the matrix-free solves.

AD limitations: topology and domain bounds are fixed; apparent-volume AD,
corrected-CRK AD, moment/selector AD, cell/cluster adjoints and higher derivatives
are not implemented. Two-bit traversal is intentionally deferred. The private
ABI requires contiguous, correctly typed/device-matched buffers and int32-sized
particle counts; it is not a public general tensor API.

## Qualification and methodology

Final V3 CPU/CUDA suite: 1,728 primal records, 288 derivative records, 144 actual
Torch backward cases and nine edge-case records, all passed. Additional checks
cover optional volume replacement, null outputs with saved state/diagnostics,
field-only derivatives, adjoint identities, aliased query/reference tensors,
saved-topology replacement, in-place topology mutation and nondefault streams.
The oracle is independent dense Torch arithmetic/autograd, not Warp or another
native implementation. Cases cover FP32/FP64, D=1/2/3, mixed periodic boundaries,
coincident points, zero supports, masks, empty sets, CSR32/64, ranges and tiles.
Compute Sanitizer memcheck reports zero errors, and racecheck reports zero
hazards/errors/warnings, on the quick CPU/CUDA suite. Instrumentation is filtered
to prototype/native kernels; it is not an unfiltered audit of Torch. HIP compiles
with hipcc for gfx942; no AMD runtime claim is made.

The scaling study is frozen V2 (before compact interpolation/prepared finalizer
cotangents): 24 fixtures, 3,168 timing records and 2,736 large-graph comparison
records. Both precisions, 1D/2D/3D; n_h=4 at 32,768, 262,144 and 1,048,576
particles; n_h=2.57 at 262,144. Actual mean neighbor counts at 262k are roughly
9/51/269 for n_h=4 and 6/22/72 for n_h=2.57. All particles use the same refined
Morton order. Warp requires CSR64; native/template CSR32 and template CSR64 are
reported separately.

Hardware: RTX PRO 6000 Blackwell (sm_120), Intel Core Ultra 9 285K, 24 Torch CPU
threads. CPU builds use g++ -O3 -march=native and Torch's stable parallel_for API;
CUDA/HIP builds use -O3 without fast-math. Existing production native binaries
were built with their prior -O2 settings and different packing/scheduling.
Therefore direct/template is the controlled abstraction comparison; production
backend comparisons include these other implementation differences.

Five randomized paired batches in the main study; seven in focused follow-ups.
GPU medians use CUDA events, with wall samples retained. CPU uses wall times.
Normal forward calls include output/reference allocation, Python call overhead
and GPU reference packing. Resident controls batch C++ calls with preallocated
outputs and packed references. Sort, radius search, transpose, oracle checks and
initial compilation are excluded. VJP timings exclude the saved primal and
transpose construction, include atomic clearing, and include finalizer
preparation/allocation where applicable. Source/binaries and individual samples
are archived, not just medians.

No other GPU compute process was observed at the activity snapshots before,
between and after fixtures/follow-ups. This is not continuous exclusive-device
reservation; graphics/system activity remains possible. Median within-case
sample range was 13.8% on GPU and 30.6% on CPU. Small percentage differences,
especially CPU launch-size cases, should not be treated as stable wins.

Seventeen Warp FP32 1D comparisons exceed the study's tolerance (all finite).
These are explicitly marked unqualified in JSON, not counted as correctness
successes. Tiny support radii relative to FP32 coordinate resolution make
geometry rounding a likely contributor, but its precise source is not isolated
here. No native comparison was waived. Do not infer numerical equivalence from
the timing of these unqualified Warp cases.

## Template overhead

Across all 168 resident operation/fixture pairs per device, median
`template/direct` time is **1.006 GPU**, **1.000 CPU**. The 10th–90th percentile
range is 0.952–1.068 GPU and 0.851–1.153 CPU. These figures indicate no systematic
abstraction penalty; they do not establish a universal 0.6% overhead. The
control uses identical operation arithmetic, reference record and schedule,
replacing only generic traversal/callback orchestration with a handwritten loop.
CPU symbol inspection finds no out-of-line generic visitor/CSR-iterator calls;
large pair routines can remain out of line in both versions. Full AVX vector
utilization was not established by this experiment.

## Forward performance

GPU milliseconds, **1,048,576 particles, 3D, FP32, n_h=4, mean 269 neighbors**.
This is the frozen V2 main study, useful for comparing the contract across
operators. Final interpolation is measured separately below.

| Operation | Template CSR32 | Direct CSR32 | Existing native/Torch | Warp CSR64 | Shared cells |
|---|---:|---:|---:|---:|---:|
| Density | 1.121 | 1.124 | 1.914 | 8.212 | 4.325 |
| Scalar interpolation | 1.208 | 1.338 | 0.997 | 10.897 | 4.688 |
| Scalar gradient | 1.342 | 1.292 | 1.953 | 9.709 | 4.978 |
| Vector Jacobian | 1.564 | 1.577 | 1.690 | 10.645 | 5.933 |
| Covariance | 1.263 | 1.313 | 1.924 | 16.485 | 5.174 |
| Normalized interpolation | 1.297 | 1.423 | 2.011 | 23.118 | 4.703 |
| CRK moments | 2.534 | 2.560 | — | 13.645 | 14.011 |

CRK moment Warp timing includes concatenating its six outputs to the common flat
output layout; the native policy writes that flat layout directly. Warp's
covariance implementation also emits its diagnostic count, whereas that output
is optional in the template timing. Normalized production/Warp baselines are
composed from two interpolations and a guarded division. These are capability
comparisons, not isolated identical-store pair benchmarks.

At this fixture the explicit CSR32 plan uses 1.141 GB and its transpose uses
another 1.141 GB. The shared-cell plan uses 32.5 MB, but this initial shared-cell
executor is much slower. This demonstrates compatibility and the storage tradeoff;
it does **not** replace or reproduce the earlier tuned clustered kernel.

![FP32 forward scaling](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-interface-20261102/forward-scaling-float32.png)

## Final operation-specific loader experiment

Plain interpolation needs V_j f_j, so the final loader can preweight that field.
The aligned 3D scalar FP32 reference record shrinks from 32 to 16 bytes. A new
paired run against frozen V2 measures the effect, with allocation/packing
included. **3D, 1M, n_h=4, milliseconds**:

| Precision | Device | Initial record | Compact record | Compact direct | Existing native/Torch |
|---|---|---:|---:|---:|---:|
| float32 | cuda | 1.295 | 1.175 | 1.160 | 0.957 |
| float32 | cpu | 88.132 | 93.930 | 94.564 | 77.611 |
| float64 | cuda | 16.710 | 16.299 | 16.016 | 15.642 |
| float64 | cpu | 86.630 | 82.181 | 80.023 | 73.722 |

The FP32 GPU improvement over the initial record is about 9%, but existing native
interpolation remains about 23% faster than the final prototype in this paired
run. Compact packing is not a universal improvement: it regresses this FP32 CPU
case and some lower-dimensional cases. This is why loaders and schedules should
remain operation/backend policies. The prototype is opt-in; these regressions do
not change production behavior.

A final width sweep (8/16/32 lanes) also finds no universal choice. For FP32,
32 lanes helps density and normalized interpolation, while 16 wins CRK moments
and the vector Jacobian. FP64 often prefers 8 lanes, particularly large moment
accumulators. No per-case winner selection was folded into main-study results.

## Adjoint ownership performance

Final GPU normalized-interpolation VJP, **3D, 1M, n_h=4, milliseconds**:

| Precision | Gradients requested | Initial owned | Final owned | Initial atomic | Final atomic |
|---|---|---:|---:|---:|---:|
| float32 | all | 7.656 | 7.574 | 14.534 | 13.857 |
| float32 | fields | 1.973 | 1.849 | 2.010 | 1.884 |
| float64 | all | 241.815 | 229.051 | 143.238 | 137.505 |
| float64 | fields | 31.831 | 29.582 | 26.970 | 24.473 |

FP32 with all input gradients benefits strongly from owned traversal (final
7.57 ms versus 13.86 ms atomic). With fields only, the difference is small.
FP64 reverses the result: recomputing pairs in two owned passes costs more than
one atomic pass on this device. CPU ownership is likewise not universally
faster. Keep both execution strategies available and qualify per workload;
CSR+transpose is a useful race-free option, not a universal speed guarantee.

Once-per-query finalizer pullback helps field-only normalized VJP by about 6–10%
in this paired run. Its speedup is smaller for dense all-input pullbacks, where
pair derivative arithmetic dominates. Explicit JVP remains separately callable
for matrix-free solves, without allocating every gradient array.

## Recommended next integration

Adopt the operation/loader/reducer/traversal/finalizer contract first for the
qualified primal operators, retaining existing tuned executors where they win.
Keep CPU raw loaders distinct from GPU packed loaders; tune query batching and
record layouts per operation. Port corrected CRK and frontend policies next,
with their manual JVP/VJP and finalizer pullbacks. Add flipped cell adjoints and
compressed traversal readers afterward. Do not replace every path with the
initial shared-cell schedule or require owned VJP unconditionally.

## Artifacts

Persistent archive:
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-interface-20261102`.
Contains frozen V2/final V3 sources and binaries, compiler commands, verification,
paired raw JSON/logs, implicit example, width sweep, sanitizer logs and plots.
The main `summary.json` has every timing record and qualification exception.
Graph paths in the archive: `forward-scaling-float32.png`,
`forward-scaling-float64.png`, `compact-interpolation-float32.png`,
`compact-interpolation-float64.png`, `vjp-ownership-float32.png`,
`vjp-ownership-float64.png`, and `finalization-vjp.png`.
