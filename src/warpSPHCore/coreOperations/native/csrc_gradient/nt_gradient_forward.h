#pragma once
#include "execution/rowNeighborhood.h"
#include "nt_gradient_operator.h"
namespace sph_native {
struct GradientRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        if (a.kernel == 0 && a.support == 11 && a.scheme == 3) {
            neighborhood::SerialSphRows<neighborhood::Gradient, neighborhood::Weight::Volume>{}(n, a, launch);
        } else {
            neighborhood::SerialSphRows<neighborhood::GeneralGradientPolicy, neighborhood::Weight::Volume>{}(n, a, launch);
        }
    }
};
} // namespace sph_native
