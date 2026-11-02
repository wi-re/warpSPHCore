#pragma once
#include "execution/sphFields.h"
#include "crk/native/csrc_correction/nt_correction.h"

namespace sph_native::neighborhood {
// This is the CRK consistency diagnostic, not ordinary SPH density. Accumulate
// its numerator and denominator together, then divide once per query.
template <class T, int D> struct ConsistencyDensity : SphOperation<T, D, 0, true> {
    using Base = SphOperation<T, D, 0, true>;
    struct Query : Base::Query { crk::Coefficients<T, D, false> coefficients; };
    using State = Sums<T, 2>;
    crk::Fields<T> coefficients;
    SPH_HD Query prepare(int64_t i) const {
        return {Base::prepare(i), crk::prepare<T, D, false>(coefficients, i)};
    }
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T dx[D];
        const T distance = q.geometry.distance(r.data, dx);
        const T w = crk::value(q.coefficients, dx, this->value_at_distance(q, edge, distance));
        const T volume = r.weight();
        sum.values[0] += this->fields.mass[edge.reference] * volume * w;
        sum.values[1] += volume * volume * w;
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        this->fields.output[q.index] = sum.values[1] == T(0) ? T(0) : sum.values[0] / sum.values[1];
    }
};
} // namespace sph_native::neighborhood
