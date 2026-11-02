#pragma once
#include "execution/sphFields.h"
#include "nt_gradient_pair.h"

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
                const T fi = q.values[c], fj = r.value(c);
                T weighted;
                if (this->scheme == 1) weighted = fj * r.weight();
                else if (this->scheme == 2) {
                    const T rhoi = this->query_density[q.index];
                    const T rhoj = this->fields.density[edge.reference];
                    weighted = this->fields.mass[edge.reference] * rhoi *
                        (fi / (rhoi * rhoi) + fj / (rhoj * rhoj));
                } else weighted = (this->scheme == 3 ? fj - fi : fj + fi) * r.weight();
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
} // namespace sph_native::neighborhood
