#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> adhesionKernel_shape(RadialJet<T> q) {
    T x = q.v;
    if (x <= T(.5))
        return {};
    T x2 = x * x, b = T(-4) * x2 + T(6) * x - T(2), c = T(2) * x2 - T(3) * x + T(1),
      p = pow(b, T(.75));
    return {T(pow(b, T(.25))), (T(6) - T(8) * x) / (T(4) * p),
            (T(16) * x2 - T(24) * x + T(11)) / (T(8) * p * c),
            T(3) * (T(3) - T(4) * x) * (T(16) * x2 - T(24) * x + T(15)) / (T(32) * c * c * p)};
}
} // namespace sph_native
