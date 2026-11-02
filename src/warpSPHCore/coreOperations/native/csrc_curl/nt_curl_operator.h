#pragma once
#include "coreOperations/native/csrc_gradient/nt_gradient_operator.h"

namespace sph_native::neighborhood {
template <class T, int D, int Fields> struct CorrectedCurl
    : CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>, true> {
    using Base = CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>, true>;
    using typename Base::Query;
    static constexpr int components = D == 1 ? 1 : D == 2 ? (Fields < 2 ? 1 : Fields / 2) : Fields;
    using State = Sums<T, components>;
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD T weighted(const Query& q, const Record& r, Neighbor edge, int c) const {
        return this->weighted_value(q, r, edge, c, this->reference_value(r, edge, c));
    }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        if constexpr (D > 1) {
            T dx[D], w, gradient[D];
            this->kernel_gradient(q, r, edge, dx, w, gradient);
            if constexpr (D == 2) {
                for (int c = 0; c < components; ++c)
                    sum.values[c] += gradient[0] * weighted(q, r, edge, components + c) -
                                     gradient[1] * weighted(q, r, edge, c);
            } else {
                constexpr int stride = Fields / 3;
                for (int c = 0; c < stride; ++c) {
                    const T x = weighted(q, r, edge, c), y = weighted(q, r, edge, stride + c),
                            z = weighted(q, r, edge, 2 * stride + c);
                    sum.values[c] += gradient[1] * z - gradient[2] * y;
                    sum.values[stride + c] += gradient[2] * x - gradient[0] * z;
                    sum.values[2 * stride + c] += gradient[0] * y - gradient[1] * x;
                }
            }
        }
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        for (int c = 0; c < components; ++c) this->fields.output[q.index * components + c] = sum.values[c];
    }
};
} // namespace sph_native::neighborhood
