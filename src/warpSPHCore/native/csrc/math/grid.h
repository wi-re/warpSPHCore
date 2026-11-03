#pragma once
#include "dataTypes/compactHash.h"
#include <cfloat>
namespace sph_native::compact {
SPH_HD inline bool finite(double value) {
    return value == value && value >= -DBL_MAX && value <= DBL_MAX;
}
template <class T, int D>
SPH_HD bool coordinates(const T *position, const Grid &grid, int64_t (&cell)[D]) {
    for (int axis = 0; axis < D; ++axis) {
        T value = position[axis];
        if (!finite(value))
            return false;
        if (grid.periodic[axis]) {
            // Use the same scalar arithmetic as minimum_image_displacement.
            const T length = T(grid.upper[axis]) - T(grid.lower[axis]);
            value -= length * ::floor((value - T(grid.lower[axis])) / length);
            const double epsilon = sizeof(T) == sizeof(float) ? FLT_EPSILON : DBL_EPSILON;
            const double slack = 32 * epsilon * double(length);
            // A large image count can exceed the scalar's wrapping precision.
            // Reject that geometry rather than index an unrelated clamped cell.
            if (!finite(value) || double(value) < grid.lower[axis] - slack ||
                double(value) > grid.upper[axis] + slack)
                return false;
        }
        const double raw = (double(value) - grid.lower[axis]) / grid.width;
        // Clamp before integer conversion, including far-outside nonperiodic
        // particles. They still undergo the exact distance predicate below.
        cell[axis] = raw <= 0 ? 0
                              : (raw >= double(grid.cells[axis] - 1) ? grid.cells[axis] - 1
                                                                     : int64_t(::floor(raw)));
    }
    return true;
}
template <int D> SPH_HD int64_t linear_key(const int64_t (&cell)[D], const Grid &grid) {
    int64_t key = 0, stride = 1;
    for (int axis = 0; axis < D; ++axis) {
        key += cell[axis] * stride;
        stride *= grid.cells[axis];
    }
    return key;
}
template <int D> SPH_HD int64_t hash_key(const int64_t (&cell)[D], const Grid &grid) {
    // Match Warp's uint32 prime hash, including its defined wraparound.
    const uint32_t primes[] = {73856093u, 19349663u, 83492791u};
    uint32_t hash = 0;
    for (int axis = 0; axis < D; ++axis)
        hash += uint32_t(cell[axis]) * (D == 1 ? 1u : primes[axis]);
    return int64_t(hash) % grid.buckets;
}
} // namespace sph_native::compact
