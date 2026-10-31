# Backend performance comparison

`bench_backends.py` qualifies and measures fresh compact-hash search and all seven
core SPH primals through Warp/Torch, native/Torch, and the standalone C++ API. It
uses CPU and CUDA by default, dimensions 1/2/3, and N = 1/16/256/4096/32768/131072/262144/524288/1000000. The
initial sweep uses float64 data, vector fields, 64-bit CSR indices/offsets and
32-bit counts. Approximately 32 other neighbors plus self (about 33 entries per particle)
keep work proportional to N once the fixture is large enough. Actual degrees are recorded.

Build the optional timing bridge from the repository root:

```bash
cmake -S . -B /tmp/warpsph-perf \
  -DWARPSPHCORE_BUILD_STANDALONE=ON \
  -DWARPSPHCORE_STANDALONE_BENCHMARKS=ON \
  -DWARPSPHCORE_NATIVE_BACKEND=cuda \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/warpsph-perf --target native_benchmark -j2

# Use the architecture of the GPU being measured; 12.0/120 is the local Blackwell.
# A native wheel can supply the Torch operators; otherwise they are built by JIT.
WARPSPHCORE_NATIVE_BUILD=gpu TORCH_CUDA_ARCH_LIST=12.0 \
python scripts/benchmarks/bench_backends.py \
  --native-library /tmp/warpsph-perf/scripts/benchmarks/native/libnative_benchmark.so \
  --output /path/to/development/core/experiments/backend-comparison/results.json

python scripts/benchmarks/analyze_backends.py \
  /path/to/development/core/experiments/backend-comparison/results.json \
  --output-dir /path/to/development/core/experiments/backend-comparison
```

For CPU only, build with `WARPSPHCORE_NATIVE_BACKEND=cpu`, omit the CUDA architecture
option, set `WARPSPHCORE_NATIVE_BUILD=cpu`, and pass `--devices cpu`. For an AMD
system, build with `WARPSPHCORE_NATIVE_BACKEND=hip` and its
`CMAKE_HIP_ARCHITECTURES`; ROCm Torch exposes GPU tensors through `--devices cuda`.
AMD timings require real hardware; offline HIP compilation is a separate check.
The bridge and standalone library link neither Torch nor Python.

A short smoke run is `--ns 16 --repeat 2 --operator-iterations 2`. Compilation and
module loading occur during untimed qualification/warmup. Each measured case
checks exact CSR membership against independently built Warp and native searches,
then checks native Torch and standalone outputs against Warp with float64
`rtol=atol=3e-9`. No backward or adjoint work is timed.

The measurement scopes are deliberately explicit:

- Search builds one fresh index, counts neighbors, scans offsets, allocates output
  and collects neighbors. Warp uses fused collection, input query order, no COO
  rows and no capacity headroom. All implementations use N hash buckets. Output
  ordering need not match; qualification canonicalizes rows outside timing.
- Operators traverse a fixed, identical, sorted CSR using Wendland2, Gather,
  Difference, Brookshaw and TrueAllToToAll. Density and covariance retain their
  usual scalar/matrix outputs. 1D curl is a degenerate zero operation.
- `standalone` allocates and frees an output for each call. `standalone_reuse`
  retains the output. Both use the normal C++ API, which performs metadata checks
  without scanning particle values or CSR contents. Descriptor setup and resident
  fixture allocation occur outside the C++ iteration loop.
- Python loops measure public calls for Warp/Torch and native/Torch. Standalone
  loops run in C++, borrowing the fixture buffers without calling Torch/Python.
  Setup in Python provides equivalent inputs; it does not make the timed C++
  execution depend on either framework.
- Completed wall time includes a final stream synchronization. Submitted time
  excludes that final wait but includes API-internal waits. It is not isolated
  kernel duration or a pure dispatch measurement. Tiny-N times bound calling
  overhead and include computation. Samples are batches of ten operator calls
  or one search by default; both batch sizes are configurable.
- Standalone fresh allocations use malloc/cudaMalloc/hipMalloc. Torch uses its
  own allocator, including the GPU caching allocator. The reuse series isolates
  the effect of retaining standalone output storage.

Below 65,536 particles, the default is five measured batches after two warmup
batches. Larger cases use one operator call per batch, three measured batches and
one warmup batch. The threshold is configurable and the actual sampling policy is
recorded per case. Raw samples,
source/build hashes, Git identity and system activity snapshots are saved in JSON
and are retained when plots show a median and min/max band. Results are saved
incrementally; the analysis tool refuses incomplete sweeps. CPU Torch threads and
standalone OpenMP rows use `--threads`, which defaults to the number of CPUs
available to the process. The original particle-count sweep explicitly used four
workers. Warp CPU kernels use serial execution. Compiler flags and CPU tuning can also differ between a JIT
extension and a standalone CMake build; preserve both build configurations when
interpreting small throughput differences.

This is a basic latency/scaling pass on fixed resident inputs. It does not cover
transfers, changing topology, index reuse, workspaces for search, Verlet lists,
CUDA graphs, corrected operators, AD, precision variants, variable supports,
multiple field ranks, isolated GPU kernel profiling, or an
entire simulation timestep. Background activity and clocks are not controlled.
Recheck close results on an idle system before making optimization decisions.
Optional input verification belongs under `tests/verify` and is never called by
any timed execution path.
Generated results belong in the external development archive described in
`docs/docs/development-artifacts.md`.

## CPU worker-count sweep

The general sweep defaults to all CPUs available to the process; use
`--threads 4` to reproduce the earlier four-worker particle-count comparison. To measure multicore scaling of the million-particle 2D case:

```bash
python scripts/benchmarks/bench_cpu_threads.py \
  --threads 4 8 16 24 \
  --native-library /tmp/warpsph-standalone-cuda/scripts/benchmarks/native/libnative_benchmark.so \
  --output-dir /path/to/development/core/experiments/cpu-thread-scaling
```

This Linux driver launches a fresh process for each worker count. It restricts
the process to the first T CPUs in its allowed affinity mask and sets
`OMP_NUM_THREADS=T`, `OMP_PLACES=cores`, and `OMP_PROC_BIND=close`. The existing
benchmark configures Torch and the standalone context to the same T workers.
Warp retains its serial CPU launch implementation. On hybrid CPUs, inspect
`manifest.json`: the first T CPUs may contain different mixtures of performance
and efficiency cores. Equal static row counts do not balance unequal core speeds.

Every process reruns the existing parity qualification outside timing. Output
contains per-worker-count JSON with raw samples and provenance, `summary.csv`,
`manifest.json` with CPU placements/topology, and `cpu-thread-scaling.png` showing
all seven primals plus fresh search. Worker counts are kept separate from the
particle-count sweep because they change the execution configuration.

## Paired interpolation precision check

`bench_interpolation_precision.py` compares FP32 and FP64 on exactly the same
2D fixture and CSR. Its timing bridge supports both precisions through private
descriptor ABI 2; rebuild `native_benchmark` after updating the harness. The
Python loader rejects an older bridge before passing a descriptor. Production
operator/library APIs are unchanged.

Prepare a deterministic fixture once, including independent exact Warp/native
search qualification, then measure each configuration in a fresh process:

```bash
python scripts/benchmarks/bench_interpolation_precision.py --prepare \
  --fixture /tmp/interpolation-precision.pt --threads 16

OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close \
 taskset --cpu-list 0-15 python scripts/benchmarks/bench_interpolation_precision.py \
  --precision float32 --device cpu --threads 16 \
  --fixture /tmp/interpolation-precision.pt \
  --native-library /tmp/warpsph-perf/scripts/benchmarks/native/libnative_benchmark.so \
  --output /path/to/development/core/experiments/precision/float32-cpu-16.json
```

Repeat with `--precision float64`, with the desired CPU placement/count, and
with `--device cuda`. All cases retain the same CSR and cast the original
float64 inputs to the selected precision. This measures resident-input
interpolation; float32 search membership is outside this check. Each case
qualifies Warp/native and standalone/native agreement plus a float64 reference
outside timing. CPU batches have one call and GPU batches twenty, with two
warmup and five measured batches. Raw error statistics and timing samples are
retained with source/build hashes and system activity snapshots.
