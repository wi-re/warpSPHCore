#pragma once
#include "execution/sphFields.h"

namespace sph_native::neighborhood {
template <class T, int D> struct Density : SphOperation<T, D, 0> {
    using Base = SphOperation<T, D, 0>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = Sums<T, 1>;
    static constexpr Weight weight = Weight::Mass;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor, State& sum) const {
        T dx[D];
        const T distance = q.geometry.distance(r.data, dx);
        sum.values[0] += r.weight() * q.kernel.value(distance);
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output) this->fields.output[q.index] = sum.values[0];
    }
};
// Uniform policy signature for shared execution adapters; density has no field.
template <class T, int D, int> using DensityPolicy = Density<T, D>;
} // namespace sph_native::neighborhood
