#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
namespace sph_native {
template <class T, int D> struct LaplacianPair {
    template <class Args>
    SPH_HD static void accumulate(const Args &a, int64_t i, int64_t j, const T (&dx)[D], T r, T w,
                                  T g, T (&sum)[D * D * D]) {
        T h = a.radius(i, j), r2 = T(0);
        for (int d = 0; d < D; ++d)
            r2 += dx[d] * dx[d];
        T factor = T(0);
        if (a.laplacian_mode == 1) {
            if (a.support == 15)
                factor = (evaluate_kernel<T, D>(dx, a.qh[i], a.kernel).laplacian +
                          evaluate_kernel<T, D>(dx, a.rh[j], a.kernel).laplacian) /
                         T(2);
            else
                factor = evaluate_kernel<T, D>(dx, h, a.kernel).laplacian;
        } else if (r > T(0)) {
            T re = r + T(1e-8) * h;
            factor = -T(2) * g * r2 / (re * (a.laplacian_mode == 4 ? r + T(1e-12) * h : re));
        }
        if (a.laplacian_mode == 3 && r > T(0)) {
            T re = r + T(1e-8) * h, F = g * r2 / (re * re);
            for (int base = 0; base < a.fields; base += D) {
                T projection = T(0);
                for (int d = 0; d < D; ++d)
                    projection += field_weight(a, i, j, base + d, true) * dx[d] / re;
                for (int d = 0; d < D; ++d)
                    sum[base + d] += (field_weight(a, i, j, base + d, true) -
                                      T(D + 2) * projection * dx[d] / re) *
                                     F;
            }
        } else
            for (int c = 0; c < a.fields; ++c)
                sum[c] += field_weight(a, i, j, c, true) * factor;
    }
};
template <class T, int D> using LaplacianRow = FieldOperatorRow<T, D, LaplacianPair<T, D>>;
struct LaplacianRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        launch(n, LaplacianRow<T, D>{a});
    }
};
} // namespace sph_native
