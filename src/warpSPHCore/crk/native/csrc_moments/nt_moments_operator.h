#pragma once
#include "execution/sphFields.h"

namespace sph_native::neighborhood {
template <class T, int D> struct Moments : SphOperation<T, D, 0> {
    using Base = SphOperation<T, D, 0>;
    using typename Base::Query;
    using typename Base::Reference;
    static constexpr int components = 1 + 2 * D + 2 * D * D + D * D * D;
    using State = CountedSums<T, components>;
    static constexpr Weight weight = Weight::Volume;
    SPH_HD State identity(const Query&) const { return {}; }
    SPH_HD void interact(const Query& q, const Reference& r, Neighbor, State& sum) const {
        T dx[D], w, g;
        const T distance = q.geometry.distance(r.data, dx);
        q.kernel.evaluate(distance, w, g);
        const T volume = r.weight();
        constexpr int m1 = 1, m2 = m1 + D, dm0 = m2 + D * D;
        constexpr int dm1 = dm0 + D, dm2 = dm1 + D * D;
        sum.values[0] += volume * w;
        for (int a = 0; a < D; ++a) {
            sum.values[m1 + a] += volume * w * dx[a];
            sum.values[dm0 + a] += volume * g * dx[a];
            for (int b = 0; b < D; ++b) {
                sum.values[m2 + a * D + b] += volume * w * dx[a] * dx[b];
                sum.values[dm1 + a * D + b] += volume * (dx[a] * g * dx[b] + (a == b ? w : T(0)));
                for (int c = 0; c < D; ++c)
                    sum.values[dm2 + c * D * D + a * D + b] += volume *
                        (g * dx[c] * dx[a] * dx[b] + w *
                         ((a == c ? dx[b] : T(0)) + (b == c ? dx[a] : T(0))));
            }
        }
        if (q.support > T(0) && distance < q.support) ++sum.count;
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output)
            for (int c = 0; c < components; ++c)
                this->fields.output[q.index * components + c] = sum.values[c];
        if (this->fields.counts) this->fields.counts[q.index] = sum.count;
    }
};
} // namespace sph_native::neighborhood
