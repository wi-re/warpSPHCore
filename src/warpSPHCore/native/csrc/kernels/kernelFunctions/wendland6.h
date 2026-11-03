#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> wendland6_shape(RadialJet<T> q) {
    auto z = positive_power(T(1), q, D == 1 ? 7 : 8);
    return z * (RadialJet<T>(T(1)) + q * T(D == 1 ? 7 : 8) + q * q * T(D == 1 ? 19 : 25) +
                q * q * q * T(D == 1 ? 21 : 32));
}
} // namespace sph_native
