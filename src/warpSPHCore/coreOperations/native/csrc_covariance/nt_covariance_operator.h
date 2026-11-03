#pragma once
#include "coreOperations/native/csrc_correction/nt_correction.h"

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
// No query/reference field is loaded for this geometric operator.
template <class T, int D, int> struct CorrectedCovariance
    : CorrectedKernel<T, D, SphOperation<T, D, 0, true>> {
    using Base = CorrectedKernel<T, D, SphOperation<T, D, 0, true>>;
    using typename Base::Query;
    using State = CountedSums<T, D * D>;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T dx[D], w, gradient[D];
        this->kernel_gradient(q, r, edge, dx, w, gradient);
        for (int a = 0; a < D; ++a)
            for (int b = 0; b < D; ++b)
                sum.values[a * D + b] -= dx[a] * r.weight() * gradient[b];
        if (this->fields.counts && crk::value(q.correction, dx, w) > T(0)) ++sum.count;
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        for (int c = 0; c < D * D; ++c) this->fields.output[q.index * D * D + c] = sum.values[c];
        if (this->fields.counts) this->fields.counts[q.index] = sum.count;
    }
};
} // namespace sph_native::neighborhood
