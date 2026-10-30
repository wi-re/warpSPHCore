#pragma once
#include "kernels/radialJet.h"
namespace sph_native {
template <class T, int D> SPH_HD RadialJet<T> gaussian_shape(RadialJet<T> q) {
    T x = q.v, e = exp(T(-128) * x * x);
    return {e, T(-256) * x * e, (T(65536) * x * x - T(256)) * e,
            (T(196608) * x - T(16777216) * x * x * x) * e};
}
} // namespace sph_native
