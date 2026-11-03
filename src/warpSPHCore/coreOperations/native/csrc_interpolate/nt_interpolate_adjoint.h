#pragma once
#include "execution/adjointTraversal.h"
namespace sph_native {
template <class T, int D> struct InterpolateAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        T rho = a.density[j], m = a.mass[j];
        for (int c = 0; c < a.fields; ++c) {
            T f = a.rvalue[j * a.fields + c];
            out.add(RVALUE, j * a.fields + c, bar[c] * m / rho * w);
            b.bw += bar[c] * f * m / rho;
            b.bm += bar[c] * f * w / rho;
            b.brr -= bar[c] * f * m * w / (rho * rho);
        }
    }
};
struct InterpolateAdjointPolicy {
    template <class T, int D> using type = InterpolateAdjointPair<T, D>;
};
} // namespace sph_native
