#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
namespace sph_native {
template <class T, int D> struct CovariancePair {
    SPH_HD static void accumulate(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D],
                                  T r, T w, T g, T (&sum)[D * D * D]) {
        for (int b = 0; b < D; ++b)
            for (int d = 0; d < D; ++d)
                sum[b * D + d] -= dx[b] * (a.mass[j] / a.density[j]) * g * dx[d];
    }
};
template <class T, int D> using CovarianceRow = FieldOperatorRow<T, D, CovariancePair<T, D>>;
struct CovarianceRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        launch(n, CovarianceRow<T, D>{a});
    }
};
} // namespace sph_native
