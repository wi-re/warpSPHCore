#pragma once
#include "dataTypes/array.h"
#include "math/distance.h"
#include "util/support.h"

namespace sph_native {
constexpr int LaplacianPositiveDivergence = 16;
template <typename T, int D> struct RowArgs {
    const T *qx, *rx, *qh, *rh, *mass, *lo, *hi;
    const bool *periodic;
    const int32_t *qkind, *rkind;
    const T *qvalue, *rvalue, *density, *qdensity;
    IndexView offsets, neighbors;
    int32_t *counts;
    int64_t *collected;
    T *output;
    int64_t references;
    int support, direction, operation, kernel;
    int fields = 1, outputs = D, scheme = 3, laplacian_mode = 2;
    bool dot_mode = false, positive_divergence = false;

    SPH_HD T displacement(int64_t i, int64_t j, T (&dx)[D]) const {
        return minimum_image_displacement<T, D>(qx + i * D, rx + j * D, lo, hi, periodic, dx);
    }
    SPH_HD T radius(int64_t i, int64_t j) const { return pairwise_support(qh[i], rh[j], support); }
    SPH_HD bool candidate(int64_t i, int64_t j) const {
        T dx[D];
        return displacement(i, j, dx) < radius(i, j);
    }
};
} // namespace sph_native
