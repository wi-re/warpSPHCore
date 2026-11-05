#pragma once
#include "execution/cooperative.cuh"
#include "execution/rowSort.h"
#include "dataTypes/array.h"
namespace sph_native {
// A row occupies one 32-lane group. Up to 512 entries use bounded register
// storage across the group; longer rows retain the in-place heap fallback.
template <int Slots,class Index> __device__ void bitonic_row(Index *values, int count, int lane) {
    using Register = std::conditional_t<sizeof(Index)==4,int,long long>;
    constexpr Register maximum = sizeof(Index)==4 ? INT32_MAX:INT64_MAX;
    Register local[Slots];
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot)
        local[slot] = lane + 32 * slot < count ? values[lane + 32 * slot] : maximum;
#pragma unroll
    for (int size = 2; size <= 32 * Slots; size *= 2) {
#pragma unroll
        for (int step = size / 2; step; step /= 2) {
            if (step >= 32) {
#pragma unroll
                for (int slot = 0; slot < Slots; ++slot) {
                    if ((slot & (step / 32)) == 0) {
                        const int partner = slot ^ (step / 32);
                        const Register a = local[slot], b = local[partner];
                        const bool ascending = ((lane + 32 * slot) & size) == 0;
                        local[slot] = ascending ? (a < b ? a : b) : (a > b ? a : b);
                        local[partner] = ascending ? (a > b ? a : b) : (a < b ? a : b);
                    }
                }
            } else {
#pragma unroll
                for (int slot = 0; slot < Slots; ++slot) {
                    const Register a = local[slot], b = group_xor<32>(a, step);
                    const int index = lane + 32 * slot;
                    const bool low = ((index & step) == 0) == ((index & size) == 0);
                    local[slot] = low ? (a < b ? a : b) : (a > b ? a : b);
                }
            }
        }
    }
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot)
        if (lane + 32 * slot < count)
            values[lane + 32 * slot] = local[slot];
}
template<class Index> struct SortCSRRow {
    IndexView offsets;
    const int32_t *counts;
    Index *indices;
    __device__ void operator()(int64_t i, int lane) const {
        const int count = counts[i];
        if (count <= 1)
            return;
        Index *row = indices + offsets[i];
        if (count <= 32)
            bitonic_row<1>(row, count, lane);
        else if (count <= 64)
            bitonic_row<2>(row, count, lane);
        else if (count <= 128)
            bitonic_row<4>(row, count, lane);
        else if (count <= 256)
            bitonic_row<8>(row, count, lane);
        else if (count <= 512)
            bitonic_row<16>(row, count, lane);
        else if (lane == 0)
            sort_row(row, count);
    }
};
template<class Index> inline void sort_csr_rows(int64_t count, IndexView offsets, const int32_t *counts, Index *indices,
                          NativeStream stream) {
    CooperativeGpuLauncher<32>{stream}(count, SortCSRRow<Index>{offsets, counts, indices});
}
} // namespace sph_native
