#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> B8_shape(RadialJet<T> q) {
    return positive_power(T(1), q, 7) - positive_power(T(.75), q, 7) * T(8) +
           positive_power(T(.5), q, 7) * T(28) - positive_power(T(.25), q, 7) * T(56);
}
} // namespace sph_native
