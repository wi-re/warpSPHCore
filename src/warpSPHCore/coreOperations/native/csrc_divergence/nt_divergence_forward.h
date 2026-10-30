#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
namespace sph_native {
template <class T, int D> struct DivergencePair {
    SPH_HD static void accumulate(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D],
                                  T r, T w, T g, T (&sum)[D * D * D]) {
        for (int c = 0; c < a.outputs; ++c)
            for (int d = 0; d < D; ++d)
                sum[c] +=
                    field_weight(a, i, j, a.dot_mode ? d * a.outputs + c : c * D + d) * g * dx[d];
    }
};
template <class T, int D> using DivergenceRow = FieldOperatorRow<T, D, DivergencePair<T, D>>;
struct DivergenceRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        launch(n, DivergenceRow<T, D>{a});
    }
};
} // namespace sph_native
