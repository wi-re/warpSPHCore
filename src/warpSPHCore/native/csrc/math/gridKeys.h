#pragma once
#include "config.h"
namespace sph_native::compact {
// Integer layout shared by native and borrowed grids. Key generation does not
// depend on real-valued bounds, width, support or their precision policies.
struct GridShape {
    int64_t cells[3]{1, 1, 1};
    int64_t buckets = 1;
};
template <int D, class Shape> SPH_HD int64_t linear_key(const int64_t (&cell)[D], const Shape &grid) {
    int64_t key = 0, stride = 1;
    for (int axis = 0; axis < D; ++axis) {
        key += cell[axis] * stride;
        stride *= grid.cells[axis];
    }
    return key;
}
template <int D, class Shape> SPH_HD int64_t hash_key(const int64_t (&cell)[D], const Shape &grid) {
    // Match Warp's uint32 prime hash, including its defined wraparound.
    const uint32_t primes[] = {73856093u, 19349663u, 83492791u};
    uint32_t hash = 0;
    for (int axis = 0; axis < D; ++axis)
        hash += uint32_t(cell[axis]) * (D == 1 ? 1u : primes[axis]);
    return int64_t(hash) % grid.buckets;
}
} // namespace sph_native::compact
