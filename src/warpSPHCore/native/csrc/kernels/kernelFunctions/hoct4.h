#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> hoct4_shape(RadialJet<T> q) {
    if (q.v <= T(.214108111463))
        return q * T(-2.154228492776) + RadialJet<T>(T(.981018558675));
    return positive_power(T(1), q, 4) + positive_power(T(.75), q, 4) * (T(16) / T(5)) -
           positive_power(T(.5), q, 4) * (T(94) / T(5));
}
} // namespace sph_native
