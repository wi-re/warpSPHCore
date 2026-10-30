#pragma once
#include "execution/adjointTraversal.h"
namespace sph_native {
template <class T, int D> struct CurlAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        if constexpr (D == 2) {
            for (int c = 0; c < a.outputs; ++c) {
                weighted_vjp(a, i, j, a.outputs + c, bar[c] * g * dx[0], false, b, out);
                weighted_vjp(a, i, j, c, -bar[c] * g * dx[1], false, b, out);
                b.bg[0] += bar[c] * field_weight(a, i, j, a.outputs + c);
                b.bg[1] -= bar[c] * field_weight(a, i, j, c);
            }
        }
        if constexpr (D == 3) {
            int S = a.fields / 3;
            for (int c = 0; c < S; ++c) {
                T u[3] = {bar[c], bar[S + c], bar[2 * S + c]},
                  f[3] = {field_weight(a, i, j, c), field_weight(a, i, j, S + c),
                          field_weight(a, i, j, 2 * S + c)};
                for (int d = 0; d < 3; ++d) {
                    int e = (d + 1) % 3, k = (d + 2) % 3;
                    weighted_vjp(a, i, j, d * S + c, g * (u[e] * dx[k] - u[k] * dx[e]), false, b,
                                 out);
                    b.bg[d] += u[k] * f[e] - u[e] * f[k];
                }
            }
        }
    }
};
struct CurlAdjointPolicy {
    template <class T, int D> using type = CurlAdjointPair<T, D>;
};
} // namespace sph_native
