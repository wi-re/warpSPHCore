#pragma once
#include "kernels/value.h"
#include "math/queryGeometry.h"
#include "util/directionality.h"
namespace sph_native {
// Shared serial/cooperative arithmetic for the common Wendland2 Gather pass.
// Components is fixed by field rank, so accumulators can stay in registers.
template <class T, int D, int Components, bool Density> struct GatherValues {
    RowArgs<T, D> a;
    template <int Stride> SPH_HD void partial(int64_t i, int lane, T (&sum)[Components]) const {
        if (!query_allowed(a.qkind[i], a.direction))
            return;
        const QueryGeometry<T, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
        const Wendland2Primal<T, D> kernel(a.qh[i]);
        const int64_t start = a.offsets[i];
        for (int32_t edge = lane; edge < a.counts[i]; edge += Stride) {
            const int64_t j = a.neighbors[start + edge];
            if (!reference_allowed(a.rkind[j], a.direction))
                continue;
            T dx[D];
            const T r = geometry.distance(a.rx + j * D, dx);
            const T w = kernel.value(r);
            if constexpr (Density) {
                sum[0] += a.mass[j] * w;
            } else {
                const T volume = a.mass[j] / a.density[j];
                for (int c = 0; c < Components; ++c)
                    sum[c] += a.rvalue[j * Components + c] * volume * w;
            }
        }
    }
    SPH_HD void operator()(int64_t i) const {
        T sum[Components] = {};
        partial<1>(i, 0, sum);
        for (int c = 0; c < Components; ++c)
            a.output[i * Components + c] = sum[c];
    }
};
} // namespace sph_native
