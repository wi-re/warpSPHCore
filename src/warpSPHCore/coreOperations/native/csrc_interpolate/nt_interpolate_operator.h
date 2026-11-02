#pragma once
#include "execution/sphFields.h"

namespace sph_native::neighborhood {

// Normalize=false matches ordinary interpolation. Normalize=true exercises a
// nonlinear finalizer over a numerator and denominator, as used by weighted
// interpolation/density operators. Both share the same traversal and AD driver.
template <class T, int D, int Fields = 1, bool Normalize = false, bool General = false>
struct Interpolate : SphOperation<T, D, Fields, General> {
    using Base = SphOperation<T, D, Fields, General>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = Sums<T, Fields + int(Normalize)>;
    static constexpr Weight weight = Normalize ? Weight::Volume : Weight::WeightedField;
    SPH_HD State identity(const Query&) const { return {}; }
    // Pull back the complete reduction once per query. Traversal never needs
    // to differentiate or duplicate this nonlinear finalization per edge.
    SPH_HD static State finalize_vjp(const State& sum, const T* cotangent) {
        State bars{};
        if constexpr (Normalize) {
            const T denominator = sum.values[Fields];
            if (denominator != T(0)) {
                for (int c = 0; c < Fields; ++c) {
                    bars.values[c] = cotangent[c] / denominator;
                    bars.values[Fields] -= cotangent[c] * sum.values[c] / (denominator * denominator);
                }
            }
        } else {
            for (int c = 0; c < Fields; ++c) bars.values[c] = cotangent[c];
        }
        return bars;
    }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        const T w = this->value(q, r, edge);
        for (int c = 0; c < Fields; ++c)
            sum.values[c] += (r.value(c) * r.weight()) * w;
        if constexpr (Normalize) sum.values[Fields] += r.weight() * w;
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.cache)
            for (int c = 0; c < Fields + int(Normalize); ++c)
                this->fields.cache[q.index * (Fields + int(Normalize)) + c] = sum.values[c];
        if (this->fields.output) {
            for (int c = 0; c < Fields; ++c) {
                T value = sum.values[c];
                if constexpr (Normalize)
                    value = sum.values[Fields] != T(0) ? value / sum.values[Fields] : T(0);
                this->fields.output[q.index * Fields + c] = value;
            }
        }
    }
};
template <class T, int D, int Fields> using InterpolatePolicy = Interpolate<T, D, Fields>;
template <class T, int D, int Fields>
using GeneralInterpolatePolicy = Interpolate<T, D, Fields, false, true>;
} // namespace sph_native::neighborhood
