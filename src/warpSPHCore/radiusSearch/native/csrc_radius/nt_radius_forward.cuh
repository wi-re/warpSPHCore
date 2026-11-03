#pragma once
#include "nt_radius_forward.h"
#include "execution/cooperative.cuh"
namespace sph_native::compact {
// Subgroup lanes share one query's cell traversal and split each cell's points.
// Ballot prefixes place accepted IDs without atomics or a scratch edge array.
template <class T, int D, int Width, bool Gather, bool Reordered = false>
struct CooperativeSearchRow {
    RowArgs<T, D> args;
    View index;
    bool collect;
    __device__ void operator()(int64_t i, int lane) const {
        if constexpr (Reordered)
            i = index.particles[i].original;
        CellTraversal<T, D> traversal(args.qx + i * D, index);
        const QueryGeometry<T, D> geometry(args.qx + i * D, args.lo, args.hi, args.periodic);
        const T query_support = args.qh[i];
        Cell cell{};
        int64_t found = 0;
        while (traversal.next(cell)) {
            for (int64_t begin = cell.begin; begin < cell.end; begin += Width) {
                const int64_t k = begin + lane;
                int64_t j = 0;
                bool accepted = false;
                if (k < cell.end) {
                    j = index.particles[k].original;
                    T dx[D];
                    const T support = Gather ? query_support : args.radius(i, j);
                    accepted = geometry.distance(args.rx + j * D, dx) < support;
                }
                const unsigned votes = group_ballot<Width>(accepted);
                if (collect && accepted) {
                    const int64_t slot = found + __popc(votes & ((1u << lane) - 1));
                    if (slot < args.counts[i])
                        args.neighbors.set(args.offsets[i] + slot,j);
                }
                found += __popc(votes);
            }
        }
        if (!collect && lane == 0)
            args.counts[i] = int32_t(found);
    }
};
} // namespace sph_native::compact
