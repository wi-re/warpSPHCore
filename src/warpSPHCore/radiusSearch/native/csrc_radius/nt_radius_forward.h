#pragma once
#include "radiusSearch/native/csrc_compact/nt_compact.h"
#include "math/queryGeometry.h"
#include "execution/rowSort.h"
namespace sph_native::compact {
template <class T, int D, bool Ordered = true> struct SearchRow {
    RowArgs<T, D> args;
    View index;
    bool collect;
    SPH_HD void operator()(int64_t i) const {
        CellTraversal<T, D> traversal(args.qx + i * D, index);
        const QueryGeometry<T, D> geometry(args.qx + i * D, args.lo, args.hi, args.periodic);
        Cell cell{};
        int64_t found = 0;
        while (traversal.next(cell)) {
            for (int64_t k = cell.begin; k < cell.end; ++k) {
                const auto j = index.particles[k].original;
                T dx[D];
                if (geometry.distance(args.rx + j * D, dx) < args.radius(i, j)) {
                    if (collect && found < args.counts[i])
                        args.collected[args.offsets[i] + found] = j;
                    ++found;
                }
            }
        }
        if (!collect)
            args.counts[i] = int32_t(found);
        else if constexpr (Ordered) {
            const auto written = found < args.counts[i] ? found : args.counts[i];
            if (written > 1)
                sort_row(args.collected + args.offsets[i], written);
        }
    }
};
} // namespace sph_native::compact
