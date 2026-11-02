#pragma once
#include "execution/sphFields.h"
#include "nt_gradient_pair.h"
#include "coreOperations/native/csrc_correction/nt_correction.h"

namespace sph_native::neighborhood {
template <class T, int D, int Fields = 1, bool General = false>
struct Gradient : SphOperation<T, D, Fields, General> {
    static constexpr bool uses_query_values = true;
    using Base = SphOperation<T, D, Fields, General>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = Sums<T, Fields * D>;
    static constexpr Weight weight = Weight::Volume;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD T weighted_value(const Query& q, const Record& r, Neighbor edge, int c, T fj) const {
        const T fi = q.values[c];
        if constexpr (!General) return (fj - fi) * r.weight();
        else {
            if (this->scheme == 1) return fj * r.weight();
            if (this->scheme == 2) {
                const T rhoi = this->query_density[q.index];
                const T rhoj = this->fields.density[edge.reference];
                return this->fields.mass[edge.reference] * rhoi *
                    (fi / (rhoi * rhoi) + fj / (rhoj * rhoj));
            }
            return (this->scheme == 3 ? fj - fi : fj + fi) * r.weight();
        }
    }
    template <class Record>
    SPH_HD FieldScales<T> field_scales(const Query& q, const Record& r, Neighbor edge) const {
        if constexpr (!General) return {-r.weight(), r.weight()};
        else {
            if (this->scheme == 1) return {T(0), r.weight()};
            if (this->scheme == 2) {
                const T rhoi = this->query_density[q.index], rhoj = this->fields.density[edge.reference];
                const T mass = this->fields.mass[edge.reference];
                return {mass / rhoi, mass * rhoi / (rhoj * rhoj)};
            }
            return {this->scheme == 3 ? -r.weight() : r.weight(), r.weight()};
        }
    }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T dx[D], w, g;
        this->evaluate(q, r, edge, dx, w, g);
        if constexpr (!General) {
            GradientPair<T, D>::template accumulate_prepared<Fields>(
                q.values, r, dx, g, sum.values);
        } else {
            // Keep the established arithmetic order, including the density
            // weighting of Symmetric gradients. No reverse-pair update occurs.
            for (int c = 0; c < Fields; ++c) {
                const T weighted = weighted_value(q, r, edge, c, r.value(c));
                for (int d = 0; d < D; ++d) sum.values[c * D + d] += weighted * g * dx[d];
            }
        }
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output)
            for (int c = 0; c < Fields * D; ++c)
                this->fields.output[q.index * Fields * D + c] = sum.values[c];
    }
};
template <class T, int D, int Fields> using GeneralGradientPolicy = Gradient<T, D, Fields, true>;

template <class T, int D, int Fields> struct CorrectedGradient
    : CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>> {
    using Base = CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>>;
    using typename Base::Query;
    using typename Base::State;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD void field_contract(const Query& q, const Record& r, Neighbor edge,
                              const T* cotangent, T* result) const {
        T dx[D], w, gradient[D];
        this->kernel_gradient(q, r, edge, dx, w, gradient);
        for (int c = 0; c < Fields; ++c)
            for (int d = 0; d < D; ++d) result[c] += cotangent[c * D + d] * gradient[d];
    }

    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T dx[D], w, gradient[D];
        this->kernel_gradient(q, r, edge, dx, w, gradient);
        for (int c = 0; c < Fields; ++c) {
            const T weighted = this->weighted_value(q, r, edge, c, this->reference_value(r, edge, c));
            for (int d = 0; d < D; ++d) sum.values[c * D + d] += weighted * gradient[d];
        }
    }
};
} // namespace sph_native::neighborhood
