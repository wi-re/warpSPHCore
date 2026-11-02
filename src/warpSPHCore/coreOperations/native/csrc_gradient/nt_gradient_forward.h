#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
#include "execution/rowNeighborhood.h"
#include "nt_gradient_pair.h"
#include "nt_gradient_operator.h"
namespace sph_native {
template <class T, int D> using GradientRow = FieldOperatorRow<T, D, GradientPair<T, D>>;
struct GradientRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        if (a.kernel == 0 && a.support == 11 && a.scheme == 3) {
            neighborhood::SerialSphRows<neighborhood::Gradient, neighborhood::Weight::Volume>{}(n, a, launch);
        } else {
            launch(n, GradientRow<T, D>{a});
        }
    }
};
} // namespace sph_native
