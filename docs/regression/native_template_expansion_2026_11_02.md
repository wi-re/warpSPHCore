# Native template coverage expansion — 2026-11-02

## Scope

The existing native density, scalar/vector/matrix interpolation and gradient
primals now use shared neighborhood operation policies across all 15 supported
kernel families, all five native support rules (Gather, Scatter, MeanSymmetric,
KernelMeanSymmetric, SuperSymmetric), and all four gradient schemes (Naive,
Symmetric, Difference, Summation). This expands template coverage of existing
native functionality; it does not change the Python API or numerical scope.

CPU uses Torch's stable-ABI parallel launcher. CUDA/HIP use the same numerical
policies with cooperative row reduction. CSR32/CSR64 neighbor IDs and offsets
remain independently selectable. The stable-ABI registrations, tensor layouts,
JIT fallback, CMake build and manual backwards are unchanged.

The Wendland2/Gather specialization remains compile-time separate from general
kernel/support evaluation. The three retained fast cases from the preceding
integration remain: CPU 3D matrix interpolation, low-degree GPU 3D FP64 vector
interpolation, and packed GPU 2D FP32 density. All other migrated cases use the
shared policy; the superseded density/interpolation/gradient row aliases and
fallback pair loops were removed. The existing standalone dispatcher now uses
the same CPU factories, to keep its build working without a second set of
primal implementations.

Only general policies carry the extra kernel/support metadata, using an empty
base for specialized policies. Specialized operation, field view and reference
loader sizes therefore retain their original 144-byte layout (FP64 3D vector
case); the general operation is 176 bytes. Loaders carry no kernel metadata.

General kernel evaluation consumes launch metadata and the original reference
ID supplied by traversal. It reads reference supports from the source array,
without enlarging packed reference records. Density/interpolation evaluate
only the radial value. General gradient weighting retains the original
multiplication order and the query/reference density terms for Symmetric
mode. Each directed interaction still accumulates only into its query's state.
Naive/Symmetric/Summation gradients also reuse the query-prepared coefficients
when the kernel is Wendland2/Gather, matching the earlier loops. No CSR scans,
CPU data readbacks, debug flags, or per-call verification options were added.

## Correctness and build qualification

- Full final CPU/CUDA Torch suite: **7,361 passed, 643 skipped**.
- Expanded CSR tests: **2,142 per device**. These include the preceding 432
  specialized cases, 1,350 kernel/support/dimension/precision cases, and 360
  gradient-scheme/rank/support/dimension/precision cases. Independent dense
  FP64 equations operate on the actual rounded inputs. Where supported, the
  same tests compare all relevant input gradients against independent Torch
  AD, including positions, supports, densities, mass, fields and periodic box.
- New general cases rotate rank, filtering and independent index/offset widths;
  the specialized cases exhaust their Cartesian product. Fixtures have
  nonuniform supports, unequal query/reference populations, disordered rows,
  unused CSR prefix storage, and coordinates several periodic images away.
  The two exhaustive new families carry the existing `extended` marker;
  ordinary focused runs can select a backend/device and use `-m "not extended"`.
- Final CPU and CUDA JIT: **2,142 passed per device** (expanded CSR tests).
- Compute Sanitizer memcheck and synccheck: **360 gradient-scheme cases each;
  zero errors** on the final implementation. Before the prepared-coefficient
  reuse fix, both tools also passed all 2,412 CSR/packed/ragged cases with zero
  errors. Filtering covers neighborhood/packed executor kernels; this is not
  an unfiltered audit of unrelated Torch or adjoint kernels.
- Python scaffolding suite: **19 passed**. Existing CPU standalone CMake
  targets: **4 CTests passed**.
- The updated CUDA wheel passes the stable-ABI symbol/RPATH audit and strict
  CPU/GPU load checks with Warp and compiler/JIT imports blocked. Its `.text`
  and `.nv_fatbin` hashes match the measured binary; compiled, packaged and
  worktree source digests match. It pins Torch **2.13.0+cu130** and targets this
  local CPU plus sm_120, not GH200.
- The three changed GPU adapters compile for HIP **gfx942**. No AMD runtime or
  ROCm wheel qualification is claimed.

PartialSymmetric remains unsupported by the native API. Reverse mode for cusp
or singular force kernels, production generic JVP/owned VJP, corrected CRK,
and migration of divergence/curl/laplacian/covariance are separate work. Their
existing support boundaries are unchanged. The new general policies are
qualified here through public CSR execution; this is not a new public cell or
compressed-neighborhood dispatch.

## Paired measurements

The final study contains **32 fixtures and 448 paired comparisons** (896
median timing records, 6,272 batch samples). Both precisions, 1D/2D/3D,
`n_h=4` and `n_h=2.57` are represented. The general cases use 262,144 particles;
the specialized control uses 32,768 and 1,048,576. Each fixture covers density,
scalar/vector/matrix interpolation and scalar/vector/matrix gradient on CPU
and GPU. Neighbors are CSR32, with Morton-sorted particles. Supports and
densities are uniform in timings; correctness fixtures separately exercise
nonuniform supports and densities.

Each immutable library runs in its own persistent worker with the native
backend explicitly selected. Measurements alternate in randomized order,
using seven batches. CPU uses all 24 threads; the GPU submitting thread is
pinned to a common core. Public allocating calls include packing and dispatch;
sorting and neighbor construction are outside the timed interval. The build
uses `-O2`, `-march=native` on CPU, CUDA sm_120, and no fast-math changes. This
is the local RTX PRO 6000 Blackwell / Core Ultra 9 285K machine, not GH200.
No unrelated GPU compute processes appeared in activity snapshots. CPU and
desktop activity was not exclusively reserved.

Median speedup ratios across each study (previous native time / migrated time;
above one is faster):

| Kernel / support / gradient scheme | GPU | CPU |
|---|---:|---:|
| Wendland2 / Gather / Difference | 1.004× | 1.004× |
| Wendland4 / KernelMeanSymmetric / Symmetric | 1.056× | 1.095× |
| CubicSpline / Scatter / Difference | 1.033× | 1.104× |
| Wendland2 / Gather / Summation | 1.008× | 0.999× |

These medians include density/interpolation controls in the Summation study;
the scheme changes only gradients. They are descriptive comparisons, not
statistical significance tests or universal speedup claims. Median full sample
ranges are about 18–20% on CPU and 9–10% on GPU. The largest sampled output
relative difference against the previous native library is 2.08e-7; independent
equation/AD tests provide the numerical qualification.

A remaining CPU control regression is explicit: at 1,048,576 particles, 3D,
FP64, `n_h=2.57`, vector gradient changes from **38.62 to 46.54 ms** in the final
sweep (about 20% longer). The dense `n_h=4` vector case is also slower
(117.71 to 133.75 ms). Earlier separate repeated samples showed the sparse
case slower too. Removing unused metadata restores specialized object sizes
but does not eliminate this timing difference. Its cause remains unresolved;
no numerical work or CSR verification was added. General Summation results
also contain slower CPU cases and one GPU comparison below a 0.9 throughput
ratio. All cases, including those regressions, remain in `summary.json`.
This completes coverage/correctness migration, not a guarantee of performance
parity for every workload.

The general policies resolve neighbor-ID width at launch and use fixed-size
rank-specific accumulators. Those changes are plausible contributors to gains
against the previous general loops; no new hardware-profiler attribution is
claimed here. The local shared library grows from 48.6 to
53.1 MiB (9.2%), including host code, device code,
and symbols for the additional general instantiations.

![Policy comparison, n_h=4](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-expansion-20261102/metadata-specialized/policy-expansion.png)

![Policy comparison, n_h=2.57](/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-expansion-20261102/metadata-specialized/policy-expansion-nh2.57.png)

Only `metadata-specialized/final-*.json` feeds these plots. Earlier exploratory
binaries and samples remain archived separately. One exploratory symmetric
run was deliberately interrupted when the prepared-coefficient reuse issue
was found; it is not included in final results.


## Reproduction

The existing public-API paired study now accepts `--kernel`, `--support`, and
`--gradient-scheme`, with the previous Wendland2/Gather/Difference defaults.
Both workers receive identical selections. These are experiment-only options;
production callers retain the existing operation properties.

```sh
python experiments/template_interface/production_study.py \
  --core /path/to/worktree --planner /path/to/exploration-v2.so \
  --baseline /path/to/previous-native.so --candidate /path/to/migrated-native.so \
  --kernel Wendland4 --support KernelMeanSymmetric --gradient-scheme Symmetric \
  --dimensions 3 --sizes 32768 262144 1048576 --output symmetric.json

python experiments/template_interface/plot_expansion.py \
  control.json symmetric.json scatter.json summation.json --output figures
```

Archive: `/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-template-expansion-20261102`. Includes immutable baseline/candidate libraries,
full source snapshot, build configuration and logs, JIT/HIP/sanitizer results,
raw paired samples, summary, graph, hardware metadata and integrity manifest.
