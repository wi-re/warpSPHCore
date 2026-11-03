#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> poly6_shape(RadialJet<T> q) {
    return jet_power(RadialJet<T>(T(1)) - q * q, 3);
}
} // namespace sph_native
