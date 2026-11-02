#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
#include "execution/valueTraversal.h"
#include "execution/rowNeighborhood.h"
#include "nt_interpolate_operator.h"
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
            launch(n, InterpolateRow<T, D>{a});
        }
    }
};
} // namespace sph_native
