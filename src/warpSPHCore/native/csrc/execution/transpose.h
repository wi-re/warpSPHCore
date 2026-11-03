#pragma once
#include "dataTypes/array.h"
#include "execution/atomic.h"
#include "execution/rowSort.h"
namespace sph_native {
struct TransposeCount {
    IndexView offsets, neighbors;
    const int32_t* counts;
    int32_t* result;
    SPH_HD void operator()(int64_t i) const {
        for (int32_t k = 0; k < counts[i]; ++k)
            add_atomic(result + neighbors[offsets[i] + k], int32_t(1));
    }
};
struct TransposeFill {
    IndexView offsets, neighbors;
    const int32_t* counts;
    const int64_t* target_offsets;
    int32_t* cursors;
    int64_t* result;
    SPH_HD void operator()(int64_t i) const {
        for (int32_t k = 0; k < counts[i]; ++k) {
            const int64_t j = neighbors[offsets[i] + k];
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            const int32_t at = atomicAdd(cursors + j, 1);
#else
            const int32_t at = __atomic_fetch_add(cursors + j, 1, __ATOMIC_RELAXED);
#endif
            result[target_offsets[j] + at] = i;
        }
    }
};
struct SortTranspose {
    const int64_t* offsets;
    const int32_t* counts;
    int64_t* neighbors;
    SPH_HD void operator()(int64_t j) const { if (counts[j] > 1) sort_row(neighbors + offsets[j], counts[j]); }
};
} // namespace sph_native
