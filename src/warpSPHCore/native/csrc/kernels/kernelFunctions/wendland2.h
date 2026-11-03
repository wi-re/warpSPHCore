#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> wendland2_shape(RadialJet<T> q) {
    auto z = positive_power(T(1), q, D == 1 ? 3 : 4);
    return z * (RadialJet<T>(T(1)) + q * T(D == 1 ? 3 : 4));
}
} // namespace sph_native

namespace sph_native {
template <class T, int D> SPH_HD void wendland(T r, T h, T &value, T &gradient) {
    value = gradient = T(0);
    if (h <= T(0) || r > h)
        return;
    auto f = wendland2_shape<T, D>({r / h, T(1)});
    const T pi = T(3.141592653589793238462643383279502884);
    const T C = D == 1 ? T(5) / T(4) : (D == 2 ? T(7) / pi : T(21) / (T(2) * pi));
    T hd = T(1);
    for (int d = 0; d < D; ++d)
        hd *= h;
    value = C * f.v / hd;
    if (r > T(0))
        gradient = C * f.d1 / (hd * h * r);
    else
        gradient = C * f.d2 / (hd * h * h);
}
} // namespace sph_native

namespace sph_native {
// Prepared primal coefficients. Gather rows share one support across neighbors;
// value and gradient evaluation then require multiplication rather than division.
template <class T, int D> struct Wendland2Primal {
    T h, inverse_h, scale, gradient_scale;
    SPH_HD explicit Wendland2Primal(T support) : h(support) {
        inverse_h = h <= T(0) ? T(0) : T(1) / h;
        T inverse_hd = T(1);
        for (int d = 0; d < D; ++d)
            inverse_hd *= inverse_h;
        const T C = D == 1 ? T(1.25) : (D == 2 ? T(2.228169203286535) : T(3.3422538049298023));
        scale = C * inverse_hd;
        gradient_scale = -T(D == 1 ? 12 : 20) * scale * inverse_h * inverse_h;
    }
    SPH_HD T value(T r) const {
        if (h <= T(0) || r > h)
            return T(0);
        const T q = r * inverse_h, t = T(1) - q;
        T factor = t * t * t;
        if constexpr (D != 1)
            factor *= t;
        return scale * factor * (T(1) + T(D == 1 ? 3 : 4) * q);
    }
    SPH_HD void evaluate(T r, T &value, T &gradient) const {
        value = gradient = T(0);
        if (h <= T(0) || r > h)
            return;
        const T q = r * inverse_h, t = T(1) - q, t2 = t * t;
        if constexpr (D == 1) {
            value = scale * t2 * t * (T(1) + T(3) * q);
            gradient = gradient_scale * t2;
        } else {
            const T t3 = t2 * t;
            value = scale * t3 * t * (T(1) + T(4) * q);
            gradient = gradient_scale * t3;
        }
    }
};
} // namespace sph_native
