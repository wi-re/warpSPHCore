#pragma once
#include "config.h"
namespace sph_native::compact {
// Read-only index views borrow device storage owned by CompactHash. Only
// occupied cells are stored; Cartesian grid volume does not allocate memory.
struct Grid {
    double lower[3]{}, upper[3]{};
    bool periodic[3]{};
    int64_t cells[3]{1, 1, 1};
    double width = 1;
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
    const int64_t *reference_keys = nullptr;
    int64_t occupied = 0;
};
} // namespace sph_native::compact
