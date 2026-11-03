#pragma once
#include "config.h"

namespace sph_native {
template <typename T> SPH_HD T pairwise_support(T hi, T hj, int mode) {
    if (mode == 11)
        return hi;
    if (mode == 12)
        return hj;
    if (mode == 13)
        return (hi + hj) / T(2);
    return hi > hj ? hi : hj;
}
} // namespace sph_native
