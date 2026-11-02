#pragma once
#include "execution/valueTraversal.h"
#include "execution/rowNeighborhood.h"
#include "nt_interpolate_operator.h"
namespace sph_native {
struct InterpolateRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        if (a.kernel == 0 && a.support == 11) {
            // The established CPU matrix loop still wins in the paired 3D
            // measurements. Keep that executor while sharing the other shapes.
            if constexpr (D == 3) {
                if (a.fields == D * D) {
                    launch(n, GatherValues<T, D, D * D, false>{a});
                    return;
                }
            }
            neighborhood::SerialSphRows<neighborhood::InterpolatePolicy, neighborhood::Weight::Volume>{}(n, a, launch);
        } else {
            neighborhood::SerialSphRows<neighborhood::GeneralInterpolatePolicy, neighborhood::Weight::Volume>{}(n, a, launch);
        }
    }
};
} // namespace sph_native
