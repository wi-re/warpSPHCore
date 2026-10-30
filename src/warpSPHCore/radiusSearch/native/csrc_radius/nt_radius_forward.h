#pragma once
#include "dataTypes/state.h"

namespace sph_native {
template <typename T, int D> struct CountRow {
    RowArgs<T, D> a;
    SPH_HD void operator()(int64_t i) const {
        int32_t count = 0;
        for (int64_t j = 0; j < a.references; ++j)
            count += a.candidate(i, j);
        a.counts[i] = count;
    }
};
template <typename T, int D> struct CollectRow {
    RowArgs<T, D> a;
    SPH_HD void operator()(int64_t i) const {
        int64_t edge = a.offsets[i];
        for (int64_t j = 0; j < a.references; ++j)
            if (a.candidate(i, j))
                a.collected[edge++] = j;
    }
};
} // namespace sph_native
