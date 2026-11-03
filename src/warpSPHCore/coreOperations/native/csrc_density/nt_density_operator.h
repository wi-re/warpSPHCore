#pragma once
#include "execution/sphFields.h"

namespace sph_native::neighborhood {
template <class T, int D, bool General = false> struct Density : SphOperation<T, D, 0, General> {
    using Base = SphOperation<T, D, 0, General>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = Sums<T, 1>;
    static constexpr Weight weight = Weight::Mass;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        sum.values[0] += r.weight() * this->value(q, r, edge);
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output) this->fields.output[q.index] = sum.values[0];
    }
};
// Uniform policy signature for shared execution adapters; density has no field.
template <class T, int D, int> using DensityPolicy = Density<T, D>;
template <class T, int D, int> using GeneralDensityPolicy = Density<T, D, true>;
} // namespace sph_native::neighborhood
