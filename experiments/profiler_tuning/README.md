# Profiler-guided interpolation tuning

This isolated experiment follows the [measured FLOP study](../../docs/regression/native_backend_flops_2026_10_31.md).
The [tuning report](../../docs/regression/native_backend_profiler_tuning_2026_10_31.md)
contains results, numerical qualifications and feature tradeoffs.

`variants.cu` includes the unchanged cluster/cell controls and adds independent
CSR gathers plus six cluster staging/image-decoding variants. `benchmark.py`
screens schedules at 262K, then times fixed choices at held-out sizes. Private
calls reuse storage; `_public` records allocate output and packing storage.
Planning, index-width conversion and correctness checks are outside timings.

```bash
nvcc -std=c++17 -O2 -arch=sm_120 -lineinfo -Xcompiler=-fPIC -shared \
  -Isrc/warpSPHCore/native/csrc experiments/profiler_tuning/variants.cu \
  -o /tmp/warpsph-profiler-tuning.so
python experiments/profiler_tuning/verify.py \
  --library /tmp/warpsph-profiler-tuning.so --output /tmp/tuning-verification.json
python experiments/profiler_tuning/benchmark.py \
  --library /tmp/warpsph-profiler-tuning.so --archive /path/to/cluster-study \
  --precision float32 --ns 262144 --screen --output /tmp/screen-float32.json
python experiments/profiler_tuning/benchmark.py \
  --library /tmp/warpsph-profiler-tuning.so --archive /path/to/cluster-study \
  --precision float32 --ns 32768 1000000 --selection /tmp/screen-float32.json \
  --native-artifact /path/to/native.so --output /tmp/final-float32.json
```

`graphs.py` measures caller-owned CUDA Graph replay on fixed topology, including
a changed-field check. `tma_gather_probe.cu` is compile-only; never launch it.
`analyze.py ARCHIVE` checks matched cases/artifacts and plots measured results.
The retained run scripts include full profiler commands and both precisions.

No operator debug argument, CSR scan or validation readback is added. The Warp
kernels and native backward/adjoint sources remain unchanged. The cluster
variants remain experimental; selected CSR changes live in the native forward
implementation and retain its generic fallbacks.
