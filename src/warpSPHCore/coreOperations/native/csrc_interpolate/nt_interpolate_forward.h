#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
namespace sph_native {
template <class T, int D> struct InterpolatePair {
    SPH_HD static void accumulate(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D],
                                  T r, T w, T g, T (&sum)[D * D * D]) {
        for (int c = 0; c < a.fields; ++c)
            sum[c] += a.rvalue[j * a.fields + c] * (a.mass[j] / a.density[j]) * w;
    }
};
template <class T, int D> using InterpolateRow = FieldOperatorRow<T, D, InterpolatePair<T, D>>;
struct InterpolateRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        launch(n, InterpolateRow<T, D>{a});
    }
};
} // namespace sph_native
