#pragma once
#include "math/distance.h"
namespace sph_native {
// Cache query coordinates and box lengths once per CSR row. Wrap relative
// displacements directly so projection does not erase close edge pairs.
template <class T, int D> struct QueryGeometry {
    T position[D], length[D];
    bool periodic[D];
    SPH_HD QueryGeometry(const T *query, const T *lo, const T *hi, const bool *flags) {
        for (int d = 0; d < D; ++d) {
            periodic[d] = flags[d];
            length[d] = hi[d] - lo[d];
            position[d] = query[d];
        }
    }
    SPH_HD T distance(const T *reference, T (&dx)[D]) const {
        T squared = T(0);
        for (int d = 0; d < D; ++d) {
            T y = reference[d];
            if (periodic[d]) {
                const T delta = position[d] - y;
                dx[d] = minimum_image_delta(delta, length[d]);
            } else {
                dx[d] = position[d] - y;
            }
            squared += dx[d] * dx[d];
        }
        return scalar_sqrt(squared);
    }
};
} // namespace sph_native
