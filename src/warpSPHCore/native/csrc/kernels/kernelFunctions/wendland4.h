#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> wendland4_shape(RadialJet<T> q) {
    auto z = positive_power(T(1), q, D == 1 ? 5 : 6);
    return z *
           (RadialJet<T>(T(1)) + q * T(D == 1 ? 5 : 6) + q * q * (D == 1 ? T(8) : T(35) / T(3)));
}
} // namespace sph_native
