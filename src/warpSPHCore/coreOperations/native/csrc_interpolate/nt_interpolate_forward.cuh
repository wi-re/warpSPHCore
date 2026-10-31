#pragma once
#include "execution/cooperative.cuh"
#include "kernels/value.h"
#include "util/directionality.h"

namespace sph_native {
template <class T, int D, int Components, int Lanes, bool WendlandGather>
struct CooperativeInterpolateRow {
    RowArgs<T, D> a;

    __device__ void operator()(int64_t i, int lane) const {
        T sum[Components] = {};
        if (query_allowed(a.qkind[i], a.direction)) {
            const int64_t start = a.offsets[i];
            for (int32_t edge = lane; edge < a.counts[i]; edge += Lanes) {
                const int64_t j = a.neighbors[start + edge];
                if (!reference_allowed(a.rkind[j], a.direction))
                    continue;
                T dx[D];
                const T r = a.displacement(i, j, dx);
                const T w = pair_kernel_value<T, D, WendlandGather>(a, i, j, r);
                const T volume = a.mass[j] / a.density[j];
#pragma unroll
                for (int c = 0; c < Components; ++c)
                    sum[c] += a.rvalue[j * Components + c] * volume * w;
            }
        }
#pragma unroll
        for (int c = 0; c < Components; ++c) {
            const T total = group_sum<Lanes>(sum[c]);
            if (lane == 0)
                a.output[i * Components + c] = total;
        }
    }
};

struct InterpolateGpuRows {
    cudaStream_t stream;

    template <class T, int D, int Components, bool WendlandGather>
    void launch(int64_t n, RowArgs<T, D> a) const {
        constexpr int lanes = 8;
        constexpr int block = 128;
        constexpr int queries_per_block = block / lanes;
        const int blocks =
            int(std::min<int64_t>((n + queries_per_block - 1) / queries_per_block, 65535));
        cooperative_rows<lanes><<<blocks, block, 0, stream>>>(
            n, CooperativeInterpolateRow<T, D, Components, lanes, WendlandGather>{a});
    }

    template <class T, int D, int Components> void fields(int64_t n, RowArgs<T, D> a) const {
        // The common Wendland2/Gather path is resolved once per launch, so its
        // neighbor loop contains no kernel-family or support-policy dispatch.
        if (a.kernel == 0 && a.support == 11)
            launch<T, D, Components, true>(n, a);
        else
            launch<T, D, Components, false>(n, a);
    }

    template <class T, int D, class UnusedLauncher>
    void operator()(int64_t n, RowArgs<T, D> a, const UnusedLauncher &) const {
        if (!n)
            return;
        if (a.fields == 1)
            fields<T, D, 1>(n, a);
        else if (a.fields == D)
            fields<T, D, D>(n, a);
        else
            fields<T, D, D * D>(n, a);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
};
} // namespace sph_native
