#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> cohesionKernel_shape(RadialJet<T> q) {
    auto p = jet_power(RadialJet<T>(T(1)) - q, 3) * jet_power(q, 3);
    return q.v < T(.5) ? p * T(2) - RadialJet<T>(T(1) / T(64)) : p;
}
} // namespace sph_native
