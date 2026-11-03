#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> spiky_shape(RadialJet<T> q) {
    return positive_power(T(1), q, 3);
}
} // namespace sph_native
