#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
namespace sph_native {
template <class T, int D> struct GradientPair {
    template <class Args>
    SPH_HD static void accumulate(const Args &a, int64_t i, int64_t j, const T (&dx)[D], T r, T w,
                                  T g, T (&sum)[D * D * D]) {
        for (int c = 0; c < a.fields; ++c)
            for (int d = 0; d < D; ++d)
                sum[c * D + d] += field_weight(a, i, j, c) * g * dx[d];
    }
};
template <class T, int D> using GradientRow = FieldOperatorRow<T, D, GradientPair<T, D>>;
struct GradientRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        launch(n, GradientRow<T, D>{a});
    }
};
} // namespace sph_native
