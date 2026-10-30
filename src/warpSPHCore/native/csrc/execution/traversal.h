#pragma once
#include "kernels/kernel.h"
#include "util/directionality.h"
namespace sph_native {
// Reusable CSR row traversal for core and frontend pair operators.
// Pair supplies components and accumulate(...); scheduling is handled separately.
template <typename T, int D, class Pair> struct OperatorRow {
    RowArgs<T, D> a;
    SPH_HD void operator()(int64_t i) const {
        const int components = Pair::components;
        for (int d = 0; d < components; ++d) a.output[i * components + d] = T(0);
        if (!query_allowed(a.qkind[i], a.direction)) return;
        const int64_t start = a.offsets[i];
        T sum[Pair::components] = {};
        for (int32_t e = 0; e < a.counts[i]; ++e) {
            const int64_t j = a.neighbors[start + e];
            if (!reference_allowed(a.rkind[j], a.direction)) continue;
            T dx[D], value, gradient;
            const T r = a.displacement(i, j, dx);
            pair_kernel(a, i, j, r, value, gradient);
            Pair::accumulate(a, i, j, dx, value, gradient, sum);
        }
        for (int d = 0; d < components; ++d) a.output[i * components + d] = sum[d];
    }
};
} // namespace sph_native
