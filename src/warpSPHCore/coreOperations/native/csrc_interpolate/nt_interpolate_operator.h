#pragma once
#include "execution/sphFields.h"
#include "coreOperations/native/csrc_correction/nt_correction.h"

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

template <class T, int D, int Fields> struct CorrectedInterpolate
    : GeneralInterpolatePolicy<T, D, Fields> {
    using Base = GeneralInterpolatePolicy<T, D, Fields>;
    using typename Base::State;
    struct Query : Base::Query { crk::Coefficients<T, D, false> correction; };
    CorrectionFields<T> corrections;

    SPH_HD Query prepare(int64_t i) const {
        return {Base::prepare(i), crk::prepare<T, D, false>(corrections.crk, i)};
    }
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD FieldScales<T> field_scales(const Query&, const Record& r, Neighbor) const {
        return {T(0), r.weight()};
    }
    template <class Record>
    SPH_HD void field_contract(const Query& q, const Record& r, Neighbor edge,
                              const T* cotangent, T* result) const {
        T dx[D];
        const T distance = q.geometry.distance(r.data, dx);
        const T raw = this->value_at_distance(q, edge, distance);
        const T weight = crk::value(q.correction, dx, raw);
        for (int c = 0; c < Fields; ++c) result[c] += cotangent[c] * weight;
    }

    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T w;
        if (corrections.crk.A) {
            T dx[D];
            const T distance = q.geometry.distance(r.data, dx);
            w = this->value_at_distance(q, edge, distance);
            w = crk::value(q.correction, dx, w);
        } else w = this->value(q, r, edge);
        for (int c = 0; c < Fields; ++c) sum.values[c] += (r.value(c) * r.weight()) * w;
    }
};
} // namespace sph_native::neighborhood
