#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> cubicSpline_shape(RadialJet<T> q) {
    return positive_power(T(1), q, 3) - positive_power(T(.5), q, 3) * T(4);
}
} // namespace sph_native
