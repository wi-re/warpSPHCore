#pragma once
#include "kernels/evaluation.h"
namespace sph_native {
template <class T, int D>
SPH_HD void kernel_vjp(const T (&dx)[D], T h, int kernel, T bw, const T (&bg)[D], T (&bx)[D],
                       T &bh) {
    if (h <= T(0))
        return;
    auto e = evaluate_kernel<T, D>(dx, h, kernel);
    for (int d = 0; d < D; ++d) {
        bx[d] += bw * e.gradient[d];
        bh += bg[d] * e.gradient_dh[d];
        for (int c = 0; c < D; ++c)
            bx[d] += e.hessian[d * D + c] * bg[c];
    }
    bh += bw * e.dh;
}
template <class T, int D> SPH_HD void laplacian_vjp(T r, T h, int kernel, T bar, T &br, T &bh) {
    const T eps = sizeof(T) == 4 ? T(1e-7) : T(1e-15);
    if (h <= T(0) || r / h < eps || r / h > T(1))
        return;
    T q = r / h, A = r + eps * h, R = r * r, C = normalization<T, D>(kernel),
      hd = power<T, D>(h, D);
    auto f = radial_shape<T, D>(q, kernel);
    T k1 = C * f.d1 / (hd * h), k2 = C * f.d2 / (hd * h * h), k3 = C * f.d3 / (hd * h * h * h);
    T s = R / (A * A), t = -R / (A * A * A) + T(D) / A;
    T ds = T(2) * r / (A * A) - T(2) * R / (A * A * A);
    T dt = -T(2) * r / (A * A * A) + T(3) * R / (A * A * A * A) - T(D) / (A * A);
    br += bar * (ds * k2 + s * k3 + dt * k1 + t * k2);
    bh += bar * (-T(2) * eps * R / (A * A * A) * k2 + s * (-T(D + 2) * k2 / h - q * k3) +
                 (T(3) * eps * R / (A * A * A * A) - T(D) * eps / (A * A)) * k1 +
                 t * (-T(D + 1) * k1 / h - q * k2));
}
} // namespace sph_native
