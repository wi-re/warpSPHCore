#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> B7_shape(RadialJet<T> q) {
    return positive_power(T(1), q, 6) - positive_power(T(5) / T(7), q, 6) * T(7) +
           positive_power(T(3) / T(7), q, 6) * T(21) - positive_power(T(1) / T(7), q, 6) * T(35);
}
} // namespace sph_native
