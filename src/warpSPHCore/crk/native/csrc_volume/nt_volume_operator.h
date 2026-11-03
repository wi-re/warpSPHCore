#pragma once
#include "execution/sphFields.h"

namespace sph_native::neighborhood {
template <class T, int D, int> struct ApparentVolume : SphOperation<T, D, 0, true> {
    using Base = SphOperation<T, D, 0, true>;
    using typename Base::Query;
    using State = Sums<T, 1>;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        sum.values[0] += this->value(q, r, edge);
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        const T total = sum.values[0];
        this->fields.output[q.index] = total == T(0) ? T(0) : T(1) / total;
    }
};
} // namespace sph_native::neighborhood
