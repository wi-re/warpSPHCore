#pragma once
#include "execution/sphFields.h"
#include "nt_gradient_pair.h"

namespace sph_native::neighborhood {
template <class T, int D, int Fields = 1>
struct Gradient : SphOperation<T, D, Fields> {
    static constexpr bool uses_query_values = true;
    using Base = SphOperation<T, D, Fields>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = Sums<T, Fields * D>;
    static constexpr Weight weight = Weight::Volume;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor, State& sum) const {
        T dx[D], w, g;
        const T distance = q.geometry.distance(r.data, dx);
        q.kernel.evaluate(distance, w, g);
        GradientPair<T, D>::template accumulate_prepared<Fields>(
            q.values, r, dx, g, sum.values);
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output)
            for (int c = 0; c < Fields * D; ++c)
                this->fields.output[q.index * Fields * D + c] = sum.values[c];
    }
};
} // namespace sph_native::neighborhood
