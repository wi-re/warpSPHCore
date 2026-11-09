#pragma once
#include "math/distance.h"
namespace sph_native {
// Cache query coordinates and domain metadata, not rounded wrapped positions.
// Direct and prepared traversal share the same precision and half-box contract.
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
            dx[d] = periodic[d] ? minimum_image_component(position[d], reference[d], length[d])
                                : position[d] - reference[d];
            squared += dx[d] * dx[d];
        }
        return scalar_sqrt(squared);
    }
};
} // namespace sph_native
