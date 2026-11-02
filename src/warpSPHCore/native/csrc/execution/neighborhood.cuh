#pragma once

#include "execution/neighborhood.h"
#include "execution/cooperative.cuh"

namespace sph_native::neighborhood {

template <class Function> __global__ void pointwise_rows(int64_t count,Function function) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
         i+=int64_t(blockDim.x)*gridDim.x) function(i);
}

template <class Function> void launch_pointwise(int64_t count,Function function,NativeStream stream) {
    if (!count) return;
    const int blocks=int(std::min<int64_t>((count+127)/128,65535));
    pointwise_rows<<<blocks,128,0,stream>>>(count,function);
    runtime::check(GPU(GetLastError)());
}

template <int Width> struct GroupReduction {
    template <class T> __device__ void sum(T& value) const {
        value = group_sum<Width>(value);
    }

    template <class T> __device__ void maximum(T& value) const {
        for (int distance = Width / 2; distance; distance /= 2) {
            const T other = group_xor<Width>(value, distance);
            value = value > other ? value : other;
        }
    }

    template <class T> __device__ void minimum(T& value) const {
        for (int distance = Width / 2; distance; distance /= 2) {
            const T other = group_xor<Width>(value, distance);
            value = value < other ? value : other;
        }
    }

    template <class Selection> __device__ void winner(Selection& value) const {
        for (int distance = Width / 2; distance; distance /= 2) {
            const auto other_value = group_xor<Width>(value.value, distance);
            const int other_index = group_xor<Width>(value.index, distance);
            if (other_index >= 0)
                value.consider(other_value, other_index);
        }
    }
};

template <int Width, class PortableRow> struct CooperativeRow {
    PortableRow row;

    __device__ void operator()(int64_t i, int lane) const {
        row.run(i, lane, Width, GroupReduction<Width>{});
    }
};

// All queries in a tile own their output for its entire source-range sweep.
// The loader's record is staged once per source tile and reused by every query.
// Inactive query groups still participate in block barriers.
template <int Width, int Queries, int References, class Operation, class Loader>
__global__ void shared_ranges(Operation operation, Loader loader,
                              const int32_t* query_begin, const int32_t* query_end,
                              Ranges source_ranges) {
    using Record = decltype(loader.load(int64_t(0)));
    __shared__ Record staged[References];
    const int task = blockIdx.x;
    const int lane = threadIdx.x % Width;
    const int group = threadIdx.x / Width;
    const int64_t i = int64_t(query_begin[task]) + group;
    const bool active = i < query_end[task];
    // prepare must be safe for a valid query; reuse the first query for padding.
    const auto query = operation.prepare(active ? i : query_begin[task]);
    auto accumulator = operation.identity(query);
    const bool allowed = active && operation.allowed(i);
    for (int range = 0; range < source_ranges.counts[task]; ++range) {
        const auto at = source_ranges.offsets[task] + range;
        const int end = source_ranges.ends[at];
        for (int start = source_ranges.begins[at]; start < end; start += References) {
            const int size = end - start < References ? end - start : References;
            for (int local = threadIdx.x; local < size; local += blockDim.x)
                staged[local] = loader.load(int64_t(start) + local);
            __syncthreads();
            if (allowed) {
                for (int local = lane; local < size; local += Width) {
                    const auto j = int64_t(start) + local;
                    if (loader.allowed(j))
                        operation.interact(query, staged[local], Neighbor{j, -1}, accumulator);
                }
            }
            __syncthreads();
        }
    }
    accumulator.reduce(GroupReduction<Width>{});
    if (active && lane == 0)
        operation.finish(query, accumulator);
}

} // namespace sph_native::neighborhood
