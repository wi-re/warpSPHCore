#pragma once
#include "execution/adjointTraversal.h"
namespace sph_native {
template <class T, int D> struct DivergenceAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        for (int c = 0; c < a.outputs; ++c)
            for (int d = 0; d < D; ++d) {
                int k = a.dot_mode ? d * a.outputs + c : c * D + d;
                weighted_vjp(a, i, j, k, bar[c] * g * dx[d], false, b, out);
                b.bg[d] += bar[c] * field_weight(a, i, j, k);
            }
    }
};
struct DivergenceAdjointPolicy {
    template <class T, int D> using type = DivergenceAdjointPair<T, D>;
};
} // namespace sph_native
