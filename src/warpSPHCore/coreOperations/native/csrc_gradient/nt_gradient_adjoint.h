#pragma once
#include "execution/adjointTraversal.h"
namespace sph_native {
template <class T, int D> struct GradientAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        for (int c = 0; c < a.fields; ++c) {
            T bf = T(0), f = field_weight(a, i, j, c);
            for (int d = 0; d < D; ++d) {
                bf += bar[c * D + d] * g * dx[d];
                b.bg[d] += bar[c * D + d] * f;
            }
            weighted_vjp(a, i, j, c, bf, false, b, out);
        }
    }
};
struct GradientAdjointPolicy {
    template <class T, int D> using type = GradientAdjointPair<T, D>;
};
} // namespace sph_native
