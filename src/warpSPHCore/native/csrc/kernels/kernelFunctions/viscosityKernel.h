#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> viscosityKernel_shape(RadialJet<T> q) {
    T x = q.v, x2 = x * x, x3 = x2 * x, x4 = x2 * x2;
    return {T(-.5) * x3 + x2 + T(.5) / x - T(1), T(-1.5) * x2 - T(.5) / x2 + T(2) * x,
            T(1) / x3 - T(3) * x + T(2), T(-3) / x4 - T(3)};
}
} // namespace sph_native
