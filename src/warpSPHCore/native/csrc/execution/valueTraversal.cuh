#pragma once
#include "execution/cooperative.cuh"
#include "execution/valueTraversal.h"
namespace sph_native {
template <class T, int D, int Components, int Lanes, bool Density, bool Fast>
struct CooperativeValues {
    RowArgs<T, D> a;
    __device__ void operator()(int64_t i, int lane) const {
        T sum[Components] = {};
        if constexpr (Fast) {
            GatherValues<T, D, Components, Density>{a}.template partial<Lanes>(i, lane, sum);
        } else if (query_allowed(a.qkind[i], a.direction)) {
            const QueryGeometry<T, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
            const int64_t start = a.offsets[i];
            for (int32_t edge = lane; edge < a.counts[i]; edge += Lanes) {
                const int64_t j = a.neighbors[start + edge];
                if (!reference_allowed(a.rkind[j], a.direction))
                    continue;
                T dx[D];
                const T r = geometry.distance(a.rx + j * D, dx);
                const T w = pair_kernel_value<T, D>(a, i, j, r);
                if constexpr (Density) {
                    sum[0] += a.mass[j] * w;
                } else {
                    const T volume = a.mass[j] / a.density[j];
#pragma unroll
                    for (int c = 0; c < Components; ++c)
                        sum[c] += a.rvalue[j * Components + c] * volume * w;
                }
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
template <bool Density> struct ValueGpuRows {
    template <class T, int D, int Components, class Launch>
    void fields(int64_t n, RowArgs<T, D> a, const Launch &launch) const {
        if (a.kernel == 0 && a.support == 11)
            launch(n, CooperativeValues<T, D, Components, Launch::width, Density, true>{a});
        else
            launch(n, CooperativeValues<T, D, Components, Launch::width, Density, false>{a});
    }
    template <class T, int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch &launch) const {
        if constexpr (Density) {
            fields<T, D, 1>(n, a, launch);
        } else {
            if (a.fields == 1)
                fields<T, D, 1>(n, a, launch);
            else if (a.fields == D)
                fields<T, D, D>(n, a, launch);
            else
                fields<T, D, D * D>(n, a, launch);
        }
    }
};
} // namespace sph_native
