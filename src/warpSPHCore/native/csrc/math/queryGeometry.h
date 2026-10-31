#pragma once
#include "math/distance.h"
namespace sph_native {
// Query-side periodic wrapping is invariant over a CSR row. Cache it once,
// retaining the same scalar arithmetic and minimum-image convention.
template <class T, int D> struct QueryGeometry {
    T position[D], lower[D], length[D];
    bool periodic[D];
    SPH_HD QueryGeometry(const T *query, const T *lo, const T *hi, const bool *flags) {
        for (int d = 0; d < D; ++d) {
            periodic[d] = flags[d];
            lower[d] = lo[d];
            length[d] = hi[d] - lo[d];
            position[d] = query[d];
            if (periodic[d]) {
                const T relative = position[d] - lower[d];
                if (!(relative >= T(0) && relative < length[d]))
                    position[d] -= length[d] * scalar_floor(relative / length[d]);
            }
        }
    }
    SPH_HD T distance(const T *reference, T (&dx)[D]) const {
        T squared = T(0);
        for (int d = 0; d < D; ++d) {
            T y = reference[d];
            if (periodic[d]) {
                const T relative = y - lower[d];
                if (!(relative >= T(0) && relative < length[d]))
                    y -= length[d] * scalar_floor(relative / length[d]);
                const T delta = position[d] - y;
                const T shifted = delta + length[d] / T(2);
                // Resolve the usual floor results 0, +1 and -1 directly.
                // Compare the same shifted scalar as the general expression,
                // preserving its half-box convention and rounding at ties.
                if (shifted >= T(0) && shifted < length[d])
                    dx[d] = delta;
                else if (length[d] > T(0) && shifted >= length[d] && shifted < T(2) * length[d])
                    dx[d] = delta - length[d];
                else if (length[d] > T(0) && shifted < T(0) && shifted >= -length[d])
                    dx[d] = delta + length[d];
                else
                    dx[d] = delta - length[d] * scalar_floor(shifted / length[d]);
            } else {
                dx[d] = position[d] - y;
            }
            squared += dx[d] * dx[d];
        }
        return scalar_sqrt(squared);
    }
};
} // namespace sph_native
