#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> quinticSpline_shape(RadialJet<T> q) {
    return positive_power(T(1), q, 5) - positive_power(T(2) / T(3), q, 5) * T(6) +
           positive_power(T(1) / T(3), q, 5) * T(15);
}
} // namespace sph_native
