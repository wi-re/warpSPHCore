#pragma once
#include "execution/traversal.h"
namespace sph_native {
template <typename T, int D> struct DifferenceGradientPair {
    static constexpr int components = D;
    SPH_HD static void accumulate(const RowArgs<T, D>& a, int64_t i, int64_t j,
                                  const T (&dx)[D], T, T gradient, T (&sum)[D]) {
        const T factor = (a.rvalue[j] - a.qvalue[i]) * (a.mass[j] / a.density[j]) * gradient;
        for (int d = 0; d < D; ++d) sum[d] += factor * dx[d];
    }
};
template <typename T, int D> using GradientRow = OperatorRow<T, D, DifferenceGradientPair<T, D>>;
struct GradientRows {
    template <typename T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher& launch) const {
        launch(n, GradientRow<T, D>{a});
    }
};
} // namespace sph_native
