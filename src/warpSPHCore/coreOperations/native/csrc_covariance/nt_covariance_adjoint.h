#pragma once
#include "execution/adjointTraversal.h"
#include "nt_covariance_kernel.h"
namespace sph_native {
template <class T, int D> struct CovarianceAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        T volume = a.mass[j] / a.density[j], bv = T(0);
        T gradient[D];
        covariance_kernel_gradient(dx,r,a.qh[i],a.rh[j],a.kernel,a.support,gradient);
        for (int c = 0; c < D; ++c)
            for (int d = 0; d < D; ++d) {
                T z = bar[c * D + d];
                b.bx[c] -= z * volume * gradient[d];
                b.bg[d] -= z * dx[c] * volume;
                bv -= z * dx[c] * gradient[d];
            }
        b.bm += bv / a.density[j];
        b.brr -= bv * a.mass[j] / (a.density[j] * a.density[j]);
    }
};
struct CovarianceAdjointPolicy {
    template <class T, int D> using type = CovarianceAdjointPair<T, D>;
};
} // namespace sph_native
