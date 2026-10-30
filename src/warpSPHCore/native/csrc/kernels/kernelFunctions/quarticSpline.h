#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> quarticSpline_shape(RadialJet<T> q) {
    return positive_power(T(1), q, 4) - positive_power(T(.6), q, 4) * T(5) +
           positive_power(T(.2), q, 4) * T(10);
}
} // namespace sph_native
