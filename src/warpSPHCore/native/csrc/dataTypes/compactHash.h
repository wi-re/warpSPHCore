#pragma once
#include "config.h"
#include <type_traits>
namespace sph_native::compact {
// Read-only index views borrow device storage owned by CompactHash. Only
// occupied cells are stored; Cartesian grid volume does not allocate memory.
template <class T> struct GridReal {
    T lower[3]{}, upper[3]{};
    T width = 1, radius_bound = 0;
};
// The active member is selected by Problem.precision before construction. This
// is source storage, not a second converted/cached copy of a double grid.
struct Grid {
    union { GridReal<float> f32; GridReal<double> f64; };
    SPH_HD Grid() : f64{} {}
    template <class T> SPH_HD auto &real() {
        if constexpr (std::is_same_v<T,float>) return f32;
        else return f64;
    }
    template <class T> SPH_HD const auto &real() const {
        if constexpr (std::is_same_v<T,float>) return f32;
        else return f64;
    }
    bool periodic[3]{};
    int64_t cells[3]{1, 1, 1};
    int64_t buckets = 1;
};
struct ParticleKey {
    int64_t bucket, cell, original;
    SPH_HD bool operator<(const ParticleKey &other) const {
        if (bucket != other.bucket)
            return bucket < other.bucket;
        if (cell != other.cell)
            return cell < other.cell;
        return original < other.original;
    }
};
struct Cell {
    int64_t key, begin, end, bucket;
};
struct Bucket {
    int64_t begin, end;
};
struct View {
    Grid grid;
    const ParticleKey *particles = nullptr;
    const Cell *cells = nullptr;
    const Bucket *buckets = nullptr;
    int64_t occupied = 0;
};
} // namespace sph_native::compact
