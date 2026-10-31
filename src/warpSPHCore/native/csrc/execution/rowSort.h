#pragma once
#include "config.h"
namespace sph_native {
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
} // namespace sph_native
