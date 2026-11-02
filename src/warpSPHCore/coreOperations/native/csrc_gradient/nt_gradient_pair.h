#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"

namespace sph_native {

// Shared by the established fallback and the prepared neighborhood policy.
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

    template <class Args>
    SPH_HD static void accumulate(const Args& a, int64_t i, int64_t j, const T (&dx)[D], T,
                                  T, T g, T (&sum)[D * D * D]) {
        for (int c = 0; c < a.fields; ++c)
            for (int d = 0; d < D; ++d)
                sum[c * D + d] += field_weight(a, i, j, c) * g * dx[d];
    }
};

} // namespace sph_native
