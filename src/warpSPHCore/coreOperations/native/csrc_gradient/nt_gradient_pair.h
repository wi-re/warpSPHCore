#pragma once
#include "config.h"

namespace sph_native {

// Prepared Difference/Gather arithmetic used by the optimized policy.
template <class T, int D> struct GradientPair {
    // Prepared Difference/Gather inputs need no general RowArgs adapter or
    // per-pair temporary accumulator. Accumulate directly into the query state
    // with a compile-time field count; retain the same multiplication order.
    template <int Fields, class Reference>
    SPH_HD static void accumulate_prepared(const T* query, const Reference& reference,
                                           const T (&dx)[D], T g, T* sum) {
        for (int c = 0; c < Fields; ++c) {
            const T weighted = (reference.value(c) - query[c]) * reference.weight() * g;
            for (int d = 0; d < D; ++d)
                sum[c * D + d] += weighted * dx[d];
        }
    }
};

} // namespace sph_native
