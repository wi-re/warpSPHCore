#pragma once
#include "kernels/radial.h"
#include "math/pow.h"
namespace sph_native {
template <class T, int D> struct KernelEvaluation {
    T value, gradient[D], hessian[D * D], dh, gradient_dh[D], laplacian;
    RadialJet<T> shape;
};
template <class T, int D>
SPH_HD KernelEvaluation<T, D> evaluate_kernel(const T *x, T h, int kernel) {
    KernelEvaluation<T, D> result{};
    if (h <= T(0))
        return result;
    T r2 = T(0);
    for (int d = 0; d < D; ++d)
        r2 += x[d] * x[d];
    T r = sqrt(r2), q = r / h;
    if (q > T(1))
        return result;
    auto f = radial_shape<T, D>(q, kernel);
    result.shape = f;
    T C = normalization<T, D>(kernel), hd = power<T, D>(h, D);
    T k1 = C * f.d1 / (hd * h), k2 = C * f.d2 / (hd * h * h);
    const T eps = sizeof(T) == 4 ? T(1e-7) : T(1e-15), rhat = r > T(0) ? r : T(1);
    result.value = C * f.v / hd;
    result.dh = -C * (T(D) * f.v + q * f.d1) / (hd * h);
    for (int a = 0; a < D; ++a) {
        result.gradient[a] = x[a] / rhat * k1;
        result.gradient_dh[a] = -x[a] / rhat * C * (T(D + 1) * f.d1 + q * f.d2) / (hd * h * h);
        for (int b = 0; b < D; ++b) {
            if (r > T(0))
                result.hessian[a * D + b] =
                    (k2 - k1 / r) * x[a] * x[b] / r2 + (a == b ? k1 / r : T(0));
            else
                result.hessian[a * D + b] = f.d1 == T(0) ? (a == b ? k2 : T(0)) : T(NAN);
        }
    }
    if (q >= eps) {
        T re = r + eps * h;
        result.laplacian = r2 / (re * re) * k2 + (-r2 / (re * re * re) + T(D) / re) * k1;
    }
    return result;
}
} // namespace sph_native
