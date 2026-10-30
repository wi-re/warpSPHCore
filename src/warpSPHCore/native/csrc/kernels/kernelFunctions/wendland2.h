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
