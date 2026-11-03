#pragma once
#include "dataTypes/array.h"
#include "execution/neighborhood.h"
namespace sph_native::neighborhood {
// A borrowed CSR with independent runtime ID/offset widths. This small
// adapter is shared infrastructure; numerical pair policies do not decode
// integer storage. Typed Csr remains available for specialized hot paths.
struct CsrView {
    IndexView offsets;
    const int32_t* counts;
    IndexView indices;

    template <class Visit>
    SPH_HD void each(int64_t row, int lane, int lanes, Visit visit) const {
        const auto begin = offsets[row];
        for (int32_t k = lane; k < counts[row]; k += lanes)
            visit(Neighbor{indices[begin + k], begin + k});
    }
};
} // namespace sph_native::neighborhood
