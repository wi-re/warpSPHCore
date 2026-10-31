#pragma once
// Shared CUDA/HIP scheduling primitives. Numerical rows own their local
// accumulators; a power-of-two group cooperates on one query particle.
#include "config.h"

namespace sph_native {
template <int Width, class T> __device__ T group_sum(T value) {
    static_assert(Width > 0 && Width <= 32 && (Width & (Width - 1)) == 0);
    for (int offset = Width / 2; offset; offset /= 2) {
#if defined(__HIP_DEVICE_COMPILE__)
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

template <int Width, class Row> __global__ void cooperative_rows(int64_t n, Row row) {
    const int lane = threadIdx.x & (Width - 1);
    const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x / Width;
    for (int64_t i = first; i < n; i += stride)
        row(i, lane);
}
} // namespace sph_native
