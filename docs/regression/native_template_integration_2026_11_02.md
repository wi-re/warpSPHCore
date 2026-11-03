# Production integration of native neighborhood templates — 2026-11-02

The qualified CSR primal policies now run through the existing production native
API, stable-ABI operator registrations, JIT build and CMake/wheel build. The
Python call signature, operator properties and backend selection are unchanged.
This work is isolated on `codex/native-cluster-pairs`, following `2af3deb`.
No frontend sorting or concurrent multi-GPU branch changes are included.

## Integrated scope

Wendland2/Gather density, scalar/vector/matrix interpolation, and Difference
gradient use the template interface on CPU and CUDA. IDs and offsets support
int32 and int64 independently; counts remain int32. Other kernels, support
schemes and gradient modes retain the established fallback implementations.
CPU 3D matrix interpolation, low-degree GPU 3D FP64 vector interpolation, and
packed 2D FP32 GPU density retain the existing executors where measurements found them faster. These are
internal dispatch choices, with no caller flag. The existing manual backwards
are unchanged and have been checked against the
new primals. The explicit JVP and owned CSR/transpose VJP remain prototype APIs;
this commit does not add public AD APIs or build transposes in ordinary calls.
Covariance, CRK moments, corrected CRK, clustered and compressed production
execution are outside this integration step.

## Implementation

Common `execution/rowNeighborhood.h/.cuh` adapt the existing borrowed `RowArgs`
once per launch. They depend on no module-owned operation: a policy is supplied
as a template argument. Policies declare field count and whether they consume a
query field. Interpolation therefore avoids loading an unused query field, and
unfiltered execution uses null kind pointers. A typed CSR resolves neighbor
width at launch; its offset accessor reads the independent offset width once
per query. These are metadata decisions, with no CSR scans or CPU readbacks.

The CPU uses `BorrowedReferenceLoader`: a source position pointer, field pointer
and scalar weight, with no padded per-edge field copy or tensor preparation
pass. The existing Torch stable parallel scheduler supplies multicore execution.
The GPU uses separately packed reference records, vector loads and the existing
operation-specific group widths. Interpolation keeps preweighted fields and
its tuned two-edge batching. The batch visitor loads independent records before
consuming them; accumulators declare `combine`, then use the normal reduction
and finalizer. Dense FP32 3D density uses 32 lanes and gradient uses 16 lanes;
lower-degree paths retain narrower groups. Selection uses storage metadata only.

The prepared gradient pair now accumulates directly into the query's fixed-size
state. It does not construct a general `RowArgs` adapter or a temporary D³ sum
for every edge. The established fallback and prepared formula remain together
in module-owned `csrc_gradient/nt_gradient_pair.h`. No public tuning/debug flags
were added. New headers are picked up by existing source manifests and package
rules; no duplicate build system or new registration was needed.

## Correctness and builds

- Full final prebuilt CPU/CUDA native Torch suite: **3,941 passed, 643 skipped**.
- New public-API matrix: **864 cases**, including independent FP64 primal and
  backward equations, both precisions, all dimensions/ranks, filtering,
  unsorted rows, unused CSR prefix, and all ID/offset-width combinations.
- Fresh CPU JIT: **433 passed**; fresh GPU JIT: **702 passed**.
- Existing large-dispatch, periodic tie, nondefault-stream, aliasing/backward,
  input-version and CUDA graph replay tests pass in the full suite.
- Compute Sanitizer memcheck and synccheck: **702 passed each, zero errors**.
  Instrumentation is filtered to the modified neighborhood kernels; this is
  not an unfiltered audit of Torch or unrelated native backwards.
- A real CUDA wheel was built with the existing packaging script and qualified
  on CPU and GPU with Warp and compiler/JIT imports blocked. Its ABI audit
  finds no direct ATen/c10/CPython symbol dependency or absolute runtime path.
  It pins Torch **2.13.0+cu130**, targets **sm_120**, and records this host's
  native CPU features. This local wheel is not a GH200/aarch64 build.
- Python scaffolding/build tests: **21 passed**. The existing standalone CPU
  target also passed its four CTest checks after the shared-header changes.
- All three modified HIP adapters compile with hipcc for **gfx942**, using
  ROCm Thrust/rocPRIM headers. No AMD runtime or ROCm wheel qualification is
  claimed; the installed Torch build is CUDA.

## Paired performance study

The final study has **48 fixtures**: FP32/FP64, D=1/2/3, N=1,024 / 32,768 /
262,144 / 1,048,576, and n_h=4 / 2.57. Seven primal operators (density, three
interpolation ranks and three gradient ranks) are timed on CPU and GPU. This
produces **672 paired comparisons, 1,344 timing records and 9,408 samples**.
There is also a separate CSR32/64 comparison at 262k in 3D, in both precisions.
The complete sweep uses frozen V4; six GPU density pairs in the figures are
replaced by a focused final follow-up for the restored 2D packed executor.
All other measured call paths are unchanged. Both raw runs and per-record
measurement provenance are retained; the focused CPU controls remain separate.
All sampled large-case output comparisons passed; small independent equations
and backwards are covered by the tests above.

The comparison uses frozen previous/native and integrated libraries in separate
persistent processes, because their Torch registrations share a namespace.
Workers run serially in randomized order, seven repeats per comparison. Idle
OpenMP workers wait passively, with `GOMP_SPINCOUNT=0` and `KMP_BLOCKTIME=0`;
CPU operations use 24 threads. GPU submission is pinned to the same logical
CPU for both workers, while CPU execution retains the full affinity. This
avoids hybrid-CPU thread migration masquerading as short-kernel overhead.

Both libraries receive identical refined Morton particle order and ordinary
CSR. Sorting, native radius search, transfers/planning and output comparisons
are outside timings. Native call allocation, packing and Python/C++ dispatch
are included. CPU uses wall time; GPU records CUDA event and wall samples.
The GPU event interval can include host submission gaps, which is why host
submission affinity matters. No other GPU compute job was observed in the
snapshots; desktop/graphics and general OS activity are not excluded.

Hardware: RTX PRO 6000 Blackwell, Intel Core Ultra 9 285K. New builds use
**-O2, no fast-math**, native CPU architecture and explicit CUDA sm_120, matching
the previous optimization level rather than relying on an O3 flag change.
CPU timings remain variable: within-case sample ranges and every repetition
are retained. Small differences should not be interpreted as universal wins.

### Representative results

Milliseconds per public call: **1,048,576 particles, 3D, n_h=4, ≈269 neighbors**.
The table comes from the final common-submission-core paired run.

| Input / operator | GPU previous | GPU integrated | CPU previous | CPU integrated |
|---|---:|---:|---:|---:|
| float32 Density | 2.031 | 1.109 | 79.053 | 65.002 |
| float32 Scalar interpolation | 1.019 | 1.058 | 85.165 | 76.167 |
| float32 Vector interpolation | 1.160 | 1.225 | 85.971 | 81.700 |
| float32 Matrix interpolation | 2.064 | 1.999 | 111.380 | 112.890 |
| float32 Scalar gradient | 1.905 | 1.336 | 123.554 | 90.053 |
| float32 Vector gradient | 1.835 | 1.546 | 180.254 | 122.964 |
| float32 Matrix gradient | 2.671 | 2.508 | 407.075 | 224.073 |
| float64 Density | 15.537 | 16.086 | 73.715 | 68.260 |
| float64 Scalar interpolation | 15.837 | 16.639 | 85.249 | 75.933 |
| float64 Vector interpolation | 16.116 | 16.726 | 100.847 | 91.753 |
| float64 Matrix interpolation | 25.040 | 25.318 | 121.117 | 117.421 |
| float64 Scalar gradient | 20.430 | 18.608 | 117.551 | 74.761 |
| float64 Vector gradient | 24.995 | 25.411 | 206.903 | 132.146 |
| float64 Matrix gradient | 41.932 | 37.734 | 434.804 | 291.755 |

The primary interpolation regression from the initial prototype is resolved
without forcing CPU and GPU into the same loader or replacing tuned GPU
batching with a uniform schedule. Dense FP32 GPU density/gradient benefit from
operation-specific scheduling, and direct prepared accumulation improves CPU
gradients. FP64 GPU gains are smaller and operation-dependent. Per-case
regressions/variation remain visible in `production-summary.json`; this is not
a claim that every shape is faster on every backend.

![Final production timings](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-production-20261102/production-3d-1m.png)

![FP32 3D scaling](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-production-20261102/production-float32-d3.png)

All 1D/2D/3D FP32/FP64 scaling figures use zero-based throughput axes and show
both neighborhood sizes. No Warp comparison is repeated here: the controlled
question is previous native versus integrated native, through the same public
API. The earlier template study retains its Warp comparison separately.

## Experiment findings that shaped the implementation

The first integration reproduced a large FP64 CPU gradient regression: the
per-pair general argument object and temporary accumulator were expensive.
Removing those objects restored scalar performance and improved larger fields.
The next run exposed CPU matrix interpolation penalties from copying source
fields into padded records per edge. Borrowed CPU field views removed that
cost. Remaining matrix/interpolation fast cases retain their established
executors rather than imposing a measured regression. These intermediate binaries/samples are retained as exploratory evidence,
not mixed into the final plots.

An unpinned short-GPU fixture showed a step from roughly 18 to 30 microseconds
in one worker, including cases with identical one-dimensional arithmetic.
Host migration/submission overhead was suspected, so the final study pins the
submitting thread to a common core. No kernel special case was introduced to
hide that measurement artifact. The benchmark still reports complete public
call intervals, not only an isolated kernel launch.

## Reproduction and artifacts

See [archived research sources](../docs/development-artifacts.md#research-sources) and
`production_study.py --help`. Existing wheel build entry point:

```sh
python scripts/packaging/build_native_wheel.py --backend cuda --gpu-arch 120 --cpu-arch native
# Run on the GH200 build node for Grace CPU + H100 instead:
python scripts/packaging/build_native_wheel.py --backend cuda --gpu-arch 90 --cpu-arch native
```

The second command is a target configuration example, not a GH200 qualification.

Archive: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-production-20261102`. Includes final/intermediate source snapshots and binaries,
JIT/CMake/compiler logs, native wheel and ABI/load checks, HIP objects, sanitizer
logs, all paired JSON/samples, CSR64 results, figures and integrity manifest.
The final data file is `production-study.json`; earlier `candidate-*` and
`final-study*` files are exploratory runs. Next integration work can expand
kernel/support coverage and corrected CRK, with the generic AD executors ported
in a separate step.
