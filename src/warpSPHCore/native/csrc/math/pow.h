#pragma once
#include "config.h"

namespace sph_native {
template <typename T, int D> SPH_HD T power(T x, int exponent) {
    T result = T(1);
    for (int k = 0; k < exponent; ++k) result *= x;
    return result;
}
} // namespace sph_native
