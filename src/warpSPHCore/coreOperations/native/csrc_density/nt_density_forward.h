#pragma once
#include "execution/rowNeighborhood.h"
#include "nt_density_operator.h"
namespace sph_native {
struct DensityRows {
    template <typename T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        if (a.kernel == 0 && a.support == 11) {
            neighborhood::SerialSphRows<neighborhood::DensityPolicy, neighborhood::Weight::Mass>{}(n, a, launch);
        } else
            neighborhood::SerialSphRows<neighborhood::GeneralDensityPolicy, neighborhood::Weight::Mass>{}(n, a, launch);
    }
};
} // namespace sph_native
