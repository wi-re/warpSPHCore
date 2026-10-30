#pragma once
#include "api.h"
#include "dataTypes/state.h"
#include "execution/atomic.h"
#include "execution/compactTraversal.h"

namespace sph_native::compact {
// Only occupied cells are stored. The conceptual Cartesian grid never allocates
// its full volume, even for sparse 3D domains.
struct Summary {
    double query_max = 0, reference_max = 0;
    int invalid = 0;
};
template <class T, int D>
SPH_HD Summary inspect_row(RowArgs<T, D> args, int64_t queries, int64_t i, const Grid &grid,
                           const int64_t *old_keys) {
    Summary result{};
    int64_t cell[D];
    if (i < queries) {
        const T support = args.qh[i];
        result.query_max = support;
        result.invalid |=
            !finite(support) || support < 0 || !coordinates<T, D>(args.qx + i * D, grid, cell);
    }
    if (i < args.references) {
        const T support = args.rh[i];
        result.reference_max = support;
        const bool valid = coordinates<T, D>(args.rx + i * D, grid, cell);
        result.invalid |= !finite(support) || support < 0 || !valid;
        if (valid && old_keys && linear_key(cell, grid) != old_keys[i])
            result.invalid |= 2;
    }
    return result;
}
SPH_HD inline Summary combine(Summary left, Summary right) {
    return {left.query_max > right.query_max ? left.query_max : right.query_max,
            left.reference_max > right.reference_max ? left.reference_max : right.reference_max,
            left.invalid | right.invalid};
}
template <class T, int D> struct KeyRow {
    RowArgs<T, D> args;
    Grid grid;
    ParticleKey *keys;
    int64_t *original_keys;
    SPH_HD void operator()(int64_t i) const {
        int64_t cell[D];
        coordinates<T, D>(args.rx + i * D, grid, cell); // validated before construction
        const auto key = linear_key(cell, grid);
        keys[i] = {hash_key(cell, grid), key, i};
        original_keys[i] = key;
    }
};
// Portable bottom-up merge sorting. Each element binary-searches its rank in the
// adjacent sorted run; original IDs make the ordering unique and deterministic.
struct MergeRow {
    const ParticleKey *input;
    ParticleKey *output;
    int64_t count, run;
    SPH_HD void operator()(int64_t i) const {
        const int64_t base = (i / (2 * run)) * (2 * run);
        const int64_t middle = base + run < count ? base + run : count;
        const int64_t end = base + 2 * run < count ? base + 2 * run : count;
        const bool left = i < middle;
        int64_t begin = left ? middle : base, stop = left ? end : middle;
        const int64_t other_begin = begin;
        while (begin < stop) {
            const int64_t mid = begin + (stop - begin) / 2;
            if (input[mid] < input[i])
                begin = mid + 1;
            else
                stop = mid;
        }
        const int64_t own_rank = i - (left ? base : middle);
        output[base + own_rank + begin - other_begin] = input[i];
    }
};
struct BoundaryRow {
    const ParticleKey *keys;
    int32_t *heads;
    SPH_HD void operator()(int64_t i) const {
        heads[i] =
            i == 0 || keys[i].bucket != keys[i - 1].bucket || keys[i].cell != keys[i - 1].cell;
    }
};
struct EmitCellRow {
    const ParticleKey *keys;
    const int32_t *heads;
    const int64_t *offsets;
    Cell *cells;
    SPH_HD void operator()(int64_t i) const {
        if (heads[i])
            cells[offsets[i]] = {keys[i].cell, i, 0, keys[i].bucket};
    }
};
struct FinishCellRow {
    Cell *cells;
    Bucket *buckets;
    int64_t occupied, particles;
    SPH_HD void operator()(int64_t i) const {
        cells[i].end = i + 1 < occupied ? cells[i + 1].begin : particles;
        const auto bucket = cells[i].bucket;
        if (i == 0 || cells[i - 1].bucket != bucket)
            buckets[bucket].begin = i;
        if (i + 1 == occupied || cells[i + 1].bucket != bucket)
            buckets[bucket].end = i + 1;
    }
};
} // namespace sph_native::compact
