# Native GPU experiments

Starting point: `0560de5`, in the isolated `codex/native-backend` worktree.
Warp remains the readable behavioral reference. No public debug or tuning flags
are added to operation calls.

## Goals and gates

1. Explain the narrow FP32/FP64 interpolation ratio with profiles of the same source.
2. Find repeatable FP32 wins over the checkpoint and compare with Warp at 32K–1M.
3. Explore radius search separately, measuring index construction and traversal.
4. Preserve FP64/CPU behavior, arbitrary periodic images, filtering, both CSR
   widths, all field ranks and manual backwards/adjoints. HIP must compile;
   AMD runtime performance remains unqualified.
5. Promote substantial, repeatable improvements; record regressions and costs.
   A 10% improvement is a useful promotion target, not an assumed outcome.

The initial CUDA micro-experiment restricts interpolation to 2D vector fields.
It borrows Torch buffers through a private timing ABI; it is not a new runtime or
public API. Production candidates must subsequently cover the normal contracts.
The driver measures CUDA events and completed wall time, qualifies every result
outside timing, rotates candidate order, and retains raw samples and activity.
Preprocessing candidates include their preprocessing launch in both intervals.

Candidates isolate lane/block scheduling, fixed metadata, narrow address indices,
reference-side coefficient preparation, unrolling, and vector loads. Reference
preparation has explicit O(N) temporary-storage costs. No approximate arithmetic
or global fast-math flag is enabled in this initial experiment.

Build with the same `-O2` as the production JIT (substitute your CUDA architecture):

```bash
nvcc -std=c++17 -O2 -arch=sm_120 -Xcompiler=-fPIC -shared \
  -I src/warpSPHCore/native/csrc experiments/native_gpu/variants.cu \
  -o /tmp/warpsph-value-variants.so
python experiments/native_gpu/bench_variants.py \
  --library /tmp/warpsph-value-variants.so \
  --fixture /tmp/warpsph-precision-million.pt --precision float32 \
  --output /path/to/development/core/experiments/native-fp32/variants.json
```

The fixture is produced by `scripts/benchmarks/bench_interpolation_precision.py
--prepare`; it contains independently qualified CSR and double-precision inputs.
Both precisions use this exact CSR. The build and results belong in the external
development archive, not Git.


## Promoted paths and costs

The final implementation packs reference positions and coefficients for the
common FP32 Wendland2/Gather primals with at least 32,768 queries. The dispatch
uses metadata only; there are no runtime CSR verification passes or debug knobs.
Other schemes, precisions, small or sparsely used reference sets retain the
existing traversal. Packed rows retain both CSR widths and manual AD behavior.
One preparation kernel and one temporary allocation are included in public-call
measurements. Typical scratch is 16–32 MB per million references; matrix fields
in 3D require 48–64 MB. The initial 1K cutoff was rejected after small-case tests.

Radius search uses four lanes per query and the existing cell order for self
searches. CSR rows still use original query IDs and sorted reference IDs. It
adds no permutation buffer or index sort. AMD support is offline compile-tested;
no AMD performance or runtime claims are made.

At 1M particles in 2D, final FP32 interpolation is 0.332 ms versus 1.487 ms at the
checkpoint and 1.493 ms for Warp. All seven FP32 primals improve 3.14–5.24× over
the checkpoint. Fresh search improves 6.15 → 2.78 ms. The detailed report is
[`docs/regression/native_backend_fp32_2026_10_31.md`](../../docs/regression/native_backend_fp32_2026_10_31.md).

## Public paired measurements

Save each compatible Torch artifact before switching source revisions. A fresh
process loads exactly one artifact. The wrapper records its SHA-256 and supplied
revision separately from the current checkout's source hashes. This historical
study uses `--neighbors 32` explicitly; future primary benchmarks default to
`--n-h 4`, with `--n-h 2.57` as comparison. It sets precision
and CSR64 before importing the package, and asserts the selected index dtypes.

```bash
# Environment: PYTHONPATH=src; WARPSPHCORE_NATIVE_BUILD=gpu; CUDA toolkit in PATH.
# Repeat for the checkpoint artifact, float64, and the dimensions of interest.
python experiments/native_gpu/bench_public.py \
  --artifact /path/to/optimized-native-torch.so --revision FINAL_COMMIT -- \
  --devices cuda --dims 2 3 --ns 32768 1000000 --precision float32 --neighbors 32 \
  --threads 16 --repeat 7 --warmup 2 --operator-iterations 10 \
  --large-case-threshold 1000001 --output /path/to/paired-packed-gpu-float32.json

# Small-case break-even checks use the same wrapper with:
# --dims 2 --ns 256 1024 4096 16384 --repeat 9 --warmup 3 --operator-iterations 50
python experiments/native_gpu/analyze_results.py /path/to/archive

# Trace actual adapter dispatch without relying on source-level expectations:
python experiments/native_gpu/profile_runtime.py --operation gradient \
  --precision float32 --n 32769 --index-bits 32 --offset-bits 64 \
  --output /path/to/gradient-mixed-width-trace.json
```

For Nsight Compute, `bench_variants.py --profile native --native-artifact PATH`
loads the specified checkpoint using the same fixture; `--profile MODE,LANES,BLOCK`
profiles a private candidate. Collect kernel metrics separately from regular
wall-time measurements and do not overlap GPU jobs.

Final raw results, graphs, source snapshots, binaries, commands, build/test logs
and profiler evidence are stored outside Git at
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/native-fp32-exploration-20261031/`.
Contended FP64 private results, rejected cutoffs and invalid early CSR-wrapper
runs are retained and explicitly excluded from promoted performance conclusions.
