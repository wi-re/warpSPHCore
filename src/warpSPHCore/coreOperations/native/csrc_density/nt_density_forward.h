#pragma once
#include "execution/traversal.h"
namespace sph_native {
template <typename T, int D> struct DensityPair {
    static constexpr int components = 1;
    SPH_HD static void accumulate(const RowArgs<T, D>& a, int64_t, int64_t j,
                                  const T (&)[D], T value, T, T (&sum)[1]) {
        sum[0] += a.mass[j] * value;
    }
};
template <typename T, int D> using DensityRow = OperatorRow<T, D, DensityPair<T, D>>;
struct DensityRows {
    template <typename T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher& launch) const {
        launch(n, DensityRow<T, D>{a});
    }
};
} // namespace sph_native
