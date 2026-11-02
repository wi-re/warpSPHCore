#pragma once
#include "execution/sphFields.h"

namespace sph_native::neighborhood {
template <class T, int D> struct Covariance : SphOperation<T, D, 0> {
    using Base = SphOperation<T, D, 0>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = CountedSums<T, D * D>;
    static constexpr Weight weight = Weight::Volume;
    SPH_HD State identity(const Query&) const { return {}; }
    SPH_HD void interact(const Query& q, const Reference& r, Neighbor, State& sum) const {
        T dx[D], w, g;
        const T distance = q.geometry.distance(r.data, dx);
        q.kernel.evaluate(distance, w, g);
        for (int a = 0; a < D; ++a)
            for (int b = 0; b < D; ++b)
                sum.values[a * D + b] += (-dx[a] * r.weight()) * (g * dx[b]);
        if (q.support > T(0) && distance < q.support) ++sum.count;
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output)
            for (int c = 0; c < D * D; ++c)
                this->fields.output[q.index * D * D + c] = sum.values[c];
        if (this->fields.counts) this->fields.counts[q.index] = sum.count;
    }
};
} // namespace sph_native::neighborhood
