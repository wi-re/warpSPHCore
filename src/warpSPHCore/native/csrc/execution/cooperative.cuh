#pragma once
// Shared CUDA/HIP scheduling primitives. Numerical rows own their local
// accumulators; a power-of-two group cooperates on one query particle.
#include "config.h"
#include "execution/device_runtime.cuh"
#include <algorithm>

namespace sph_native {
template <int Width, class T> __device__ T group_sum(T value) {
    static_assert(Width > 0 && Width <= 32 && (Width & (Width - 1)) == 0);
    for (int offset = Width / 2; offset; offset /= 2) {
#if defined(__HIPCC__) || defined(SPH_NATIVE_HIP) || defined(USE_ROCM)
        value += __shfl_down(value, offset, Width);
#else
        // Every non-exited lane in this group reaches the reduction, including
        // lanes whose ragged CSR row assigned them no neighbor contributions.
        const unsigned first = (threadIdx.x & 31) & ~(Width - 1);
        const unsigned mask = (0xffffffffu >> (32 - Width)) << first;
        value += __shfl_down_sync(mask, value, offset, Width);
#endif
    }
    return value;
}

template <int Width, class T> __device__ T group_xor(T value, int partner) {
#if defined(__HIPCC__) || defined(SPH_NATIVE_HIP) || defined(USE_ROCM)
    return __shfl_xor(value, partner, Width);
#else
    const unsigned first = (threadIdx.x & 31) & ~(Width - 1);
    const unsigned mask = (0xffffffffu >> (32 - Width)) << first;
    return __shfl_xor_sync(mask, value, partner, Width);
#endif
}

template <int Width, class Row> __global__ void cooperative_rows(int64_t n, Row row) {
    const int lane = threadIdx.x & (Width - 1);
    const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x / Width;
    for (int64_t i = first; i < n; i += stride)
        row(i, lane);
}
template <int Width> struct CooperativeGpuLauncher {
    static constexpr int width = Width;
    NativeStream stream;
    template <class Row> void operator()(int64_t n, Row row) const {
        if (!n)
            return;
        constexpr int block = 128, per_block = block / Width;
        const int blocks = int(std::min<int64_t>((n + per_block - 1) / per_block, 65535));
        cooperative_rows<Width><<<blocks, block, 0, stream>>>(n, row);
        runtime::check(GPU(GetLastError)());
    }
};
} // namespace sph_native
