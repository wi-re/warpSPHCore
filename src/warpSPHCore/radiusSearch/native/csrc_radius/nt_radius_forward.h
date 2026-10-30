#pragma once
#include "radiusSearch/native/csrc_compact/nt_compact.h"
namespace sph_native::compact {
SPH_HD inline void mark_error(int32_t *error) {
    if (!error)
        return;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicOr(error, 1);
#else
    __atomic_fetch_or(error, 1, __ATOMIC_RELAXED);
#endif
}
// Keep the existing native CSR contract: each row contains original reference
// IDs in ascending order. Heap sorting needs no per-row scratch allocation.
SPH_HD inline void sift_down(int64_t *values, int64_t root, int64_t count) {
    while (root < count / 2) {
        int64_t child = 2 * root + 1;
        if (child + 1 < count && values[child] < values[child + 1])
            ++child;
        if (values[root] >= values[child])
            break;
        const auto saved = values[root];
        values[root] = values[child];
        values[child] = saved;
        root = child;
    }
}
SPH_HD inline void sort_row(int64_t *values, int64_t count) {
    for (int64_t root = count / 2; root > 0; --root)
        sift_down(values, root - 1, count);
    for (int64_t end = count; end > 1; --end) {
        const auto saved = values[0];
        values[0] = values[end - 1];
        values[end - 1] = saved;
        sift_down(values, 0, end - 1);
    }
}
template <class T, int D> struct SearchRow {
    RowArgs<T, D> args;
    View index;
    bool collect;
    int32_t *error = nullptr;
    SPH_HD void operator()(int64_t i) const {
        CellTraversal<T, D> traversal(args.qx + i * D, index);
        Cell cell{};
        int64_t found = 0;
        while (traversal.next(cell)) {
            for (int64_t k = cell.begin; k < cell.end; ++k) {
                const auto j = index.particles[k].original;
                if (args.candidate(i, j)) {
                    if (collect && found < args.counts[i])
                        args.collected[args.offsets[i] + found] = j;
                    ++found;
                }
            }
        }
        if (!collect)
            args.counts[i] = int32_t(found);
        else {
            if (found != args.counts[i])
                mark_error(error);
            const auto written = found < args.counts[i] ? found : args.counts[i];
            if (written > 1)
                sort_row(args.collected + args.offsets[i], written);
        }
    }
};
} // namespace sph_native::compact
