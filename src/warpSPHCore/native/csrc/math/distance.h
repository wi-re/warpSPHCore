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

// Truncated remainder keeps local displacement bits; half-box recentering does not.
template <class T> SPH_HD T scalar_mod(T value, T length) {
    if constexpr (sizeof(T) == sizeof(float))
        return ::fmodf(value, length);
    else
        return ::fmod(value, length);
}
template <class T> SPH_HD T minimum_image_component(T x, T y, T length) {
    T delta = x - y;
    const T half = length / T(2);
    if (delta >= half || delta < -half) {
        // Separate remainders retain y even if a very large image-shifted x
        // would round it away in the subtraction. No domain-origin shift.
        delta = scalar_mod(scalar_mod(x, length) - scalar_mod(y, length), length);
        if (delta >= half) delta -= length;
        else if (delta < -half) delta += length;
    }
    return delta;
}

template <typename T, int D>
SPH_HD T minimum_image_displacement(const T *qx, const T *rx, const T *lo, const T *hi,
                                    const bool *periodic, T (&dx)[D]) {
    T rr = T(0);
    for (int d = 0; d < D; ++d) {
        dx[d] = periodic[d] ? minimum_image_component(qx[d], rx[d], hi[d] - lo[d])
                            : qx[d] - rx[d];
        rr += dx[d] * dx[d];
    }
    return scalar_sqrt(rr);
}
} // namespace sph_native
