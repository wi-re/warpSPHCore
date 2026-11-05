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

// Wrap the displacement without adding and then subtracting half a box.
// This preserves close-pair separations and opposite-pair symmetry away from
// the unavoidable half-box cut (+half maps to -half).
template <class T> SPH_HD T minimum_image_delta(T delta, T length) {
    const T half = length / T(2);
    if (delta >= -half && delta < half) return delta;
    if (delta >= half && delta < length) return delta - length;
    if (delta < -half && delta >= -length) return delta + length;
    return delta - length * scalar_floor((delta + half) / length);
}

template <typename T, int D>
SPH_HD T minimum_image_displacement(const T *qx, const T *rx, const T *lo, const T *hi,
                                    const bool *periodic, T (&dx)[D]) {
    T rr = T(0);
    for (int d = 0; d < D; ++d) {
        T x = qx[d], y = rx[d];
        if (periodic[d]) {
            const T length = hi[d] - lo[d];
            T delta = x - y;
            dx[d] = minimum_image_delta(delta, length);
        } else {
            dx[d] = x - y;
        }
        rr += dx[d] * dx[d];
    }
    return scalar_sqrt(rr);
}
} // namespace sph_native
