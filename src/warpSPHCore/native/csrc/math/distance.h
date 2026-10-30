#pragma once
#include "config.h"

namespace sph_native {
template <typename T, int D>
SPH_HD T minimum_image_displacement(const T *qx, const T *rx, const T *lo, const T *hi,
                                    const bool *periodic, T (&dx)[D]) {
    T rr = T(0);
    for (int d = 0; d < D; ++d) {
        T x = qx[d], y = rx[d];
        if (periodic[d]) {
            const T length = hi[d] - lo[d];
            x -= length * ::floor((x - lo[d]) / length);
            y -= length * ::floor((y - lo[d]) / length);
            T delta = x - y;
            dx[d] = delta - length * ::floor((delta + length / T(2)) / length);
        } else {
            dx[d] = x - y;
        }
        rr += dx[d] * dx[d];
    }
    return ::sqrt(rr);
}
} // namespace sph_native
