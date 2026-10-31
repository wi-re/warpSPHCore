#!/usr/bin/env bash
# Run from any directory. Supply the output archive and prior control artifacts.
set -euo pipefail
compression_archive=$(realpath -m "${1:?output archive required}")
compression_controls=$(realpath "${2:?prior profiler artifact directory required}")
compression_root=$(cd "$(dirname "$0")/../.." && pwd)
compression_python=${PYTHON:-python}
compression_nvcc=${NVCC:-nvcc}
compression_arch=${CUDA_ARCH:-sm_120}
cd "$compression_root"
mkdir -p "$compression_archive/artifacts" "$compression_archive/measured-source"
cp experiments/csr_compression/*.py experiments/csr_compression/kernels.cu \
  experiments/csr_compression/README.md "$compression_archive/measured-source/"
"$compression_nvcc" -std=c++17 -O2 -arch="$compression_arch" -lineinfo -Xcompiler=-fPIC -shared \
  -Isrc/warpSPHCore/native/csrc experiments/csr_compression/kernels.cu \
  -o "$compression_archive/artifacts/compression.so"
"$compression_python" experiments/csr_compression/verify.py \
  --library "$compression_archive/artifacts/compression.so" --output "$compression_archive/verification.json"
for compression_precision in float32 float64; do
  "$compression_python" experiments/csr_compression/benchmark.py \
    --library "$compression_archive/artifacts/compression.so" --artifacts "$compression_controls" \
    --precision "$compression_precision" --screen --ns 262144 --repeat 7 --iterations 5 \
    --output "$compression_archive/screen-$compression_precision.json"
  "$compression_python" experiments/csr_compression/benchmark.py \
    --library "$compression_archive/artifacts/compression.so" --artifacts "$compression_controls" \
    --precision "$compression_precision" --selection "$compression_archive/screen-$compression_precision.json" \
    --ns 32768 131072 524288 1000000 --repeat 9 --iterations 5 \
    --output "$compression_archive/final-$compression_precision.json"
done
"$compression_python" experiments/csr_compression/analyze.py "$compression_archive"
