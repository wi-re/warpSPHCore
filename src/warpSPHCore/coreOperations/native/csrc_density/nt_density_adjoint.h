#pragma once
#include "execution/adjointTraversal.h"
namespace sph_native {
template <class T, int D> struct DensityAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        b.bw += bar[0] * a.mass[j];
        b.bm += bar[0] * w;
    }
};
struct DensityAdjointPolicy {
    template <class T, int D> using type = DensityAdjointPair<T, D>;
};
} // namespace sph_native
