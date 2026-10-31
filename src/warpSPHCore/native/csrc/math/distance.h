#pragma once
#include "config.h"

namespace sph_native {
// Global C math overloads promote float to double on ordinary C++ compilers.
// Select the scalar operation explicitly for the requested tensor precision.
template <class T> SPH_HD T scalar_floor(T value) {
    if constexpr (sizeof(T) == sizeof(float))
        return ::floorf(value);
    else
        return ::floor(value);
}
template <class T> SPH_HD T scalar_sqrt(T value) {
    if constexpr (sizeof(T) == sizeof(float))
        return ::sqrtf(value);
    else
        return ::sqrt(value);
}

template <typename T, int D>
SPH_HD T minimum_image_displacement(const T *qx, const T *rx, const T *lo, const T *hi,
                                    const bool *periodic, T (&dx)[D]) {
    T rr = T(0);
    for (int d = 0; d < D; ++d) {
        T x = qx[d], y = rx[d];
        if (periodic[d]) {
            const T length = hi[d] - lo[d];
            x -= length * scalar_floor((x - lo[d]) / length);
            y -= length * scalar_floor((y - lo[d]) / length);
            T delta = x - y;
            dx[d] = delta - length * scalar_floor((delta + length / T(2)) / length);
        } else {
            dx[d] = x - y;
        }
        rr += dx[d] * dx[d];
    }
    return scalar_sqrt(rr);
}
} // namespace sph_native
