#pragma once
#include "dataTypes/compactHash.h"
#include <cfloat>
namespace sph_native::compact {
template <class T> SPH_HD inline bool finite(T value) {
    constexpr T maximum = sizeof(T) == sizeof(float) ? T(FLT_MAX) : T(DBL_MAX);
    return value == value && value >= -maximum && value <= maximum;
}
template <class T> SPH_HD inline T floor_scalar(T value) {
    if constexpr (sizeof(T) == sizeof(float)) return ::floorf(value);
    else return ::floor(value);
}
template <class T, int D>
SPH_HD bool coordinates(const T *position, const Grid &grid, int64_t (&cell)[D]) {
    const auto &real = grid.real<T>();
    for (int axis = 0; axis < D; ++axis) {
        T value = position[axis];
        if (!finite(value)) return false;
        if (grid.periodic[axis]) {
            const T length = real.upper[axis] - real.lower[axis];
            value -= length * floor_scalar((value - real.lower[axis]) / length);
            constexpr T epsilon = sizeof(T) == sizeof(float) ? T(FLT_EPSILON) : T(DBL_EPSILON);
            const T slack = T(32) * epsilon * length;
            if (!finite(value) || value < real.lower[axis] - slack || value > real.upper[axis] + slack)
                return false;
        }
        const T raw = (value - real.lower[axis]) / real.width;
        cell[axis] = raw <= T(0) ? 0 : raw >= T(grid.cells[axis] - 1) ? grid.cells[axis] - 1 : int64_t(floor_scalar(raw));
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
