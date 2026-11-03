#pragma once
#include "execution/cooperative.cuh"
#include "execution/rowSort.h"
#include "dataTypes/array.h"
namespace sph_native {
// A row occupies one 32-lane group. Common degrees fit one or two register
// values per lane; longer rows retain the general in-place heap fallback.
template <bool Wide,class Index> __device__ void bitonic_row(Index *values, int count, int lane) {
    using Register = std::conditional_t<sizeof(Index)==4,int,long long>;
    constexpr Register maximum = sizeof(Index)==4 ? INT32_MAX:INT64_MAX;
    Register first = lane < count ? values[lane] : maximum;
    Register second = maximum;
    if constexpr (Wide)
        second = lane + 32 < count ? values[lane + 32] : maximum;
#pragma unroll
    for (int size = 2; size <= (Wide ? 64 : 32); size *= 2) {
#pragma unroll
        for (int step = size / 2; step; step /= 2) {
            Register partner_first, partner_second = maximum;
            if (step == 32) {
                partner_first = second;
                partner_second = first;
            } else {
                partner_first = group_xor<32>(first, step);
                if constexpr (Wide)
                    partner_second = group_xor<32>(second, step);
            }
            const bool low = ((lane & step) == 0) == ((lane & size) == 0);
            first = low ? (first < partner_first ? first : partner_first)
                        : (first > partner_first ? first : partner_first);
            if constexpr (Wide) {
                const int index = lane + 32;
                const bool low_second = ((index & step) == 0) == ((index & size) == 0);
                second = low_second ? (second < partner_second ? second : partner_second)
                                    : (second > partner_second ? second : partner_second);
            }
        }
    }
    if (lane < count)
        values[lane] = first;
    if constexpr (Wide) {
        if (lane + 32 < count)
            values[lane + 32] = second;
    }
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
            bitonic_row<false>(row, count, lane);
        else if (count <= 64)
            bitonic_row<true>(row, count, lane);
        else if (lane == 0)
            sort_row(row, count);
    }
};
template<class Index> inline void sort_csr_rows(int64_t count, IndexView offsets, const int32_t *counts, Index *indices,
                          NativeStream stream) {
    CooperativeGpuLauncher<32>{stream}(count, SortCSRRow<Index>{offsets, counts, indices});
}
} // namespace sph_native
