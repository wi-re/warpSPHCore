#pragma once
#include "coreOperations/native/csrc_gradient/nt_gradient_operator.h"
#include "kernels/evaluation.h"

namespace sph_native::neighborhood {
template <class T, int D, int Fields> struct CorrectedLaplacian
    : CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>, true> {
    using Base = CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>, true>;
    using typename Base::Query;
    using State = Sums<T, Fields>;
    int mode = 2;
    template <class Args> void configure(const Args& a) {
        Base::configure(a);
        mode = a.laplacian_mode;
    }
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD T difference(const Query& q, const Record& r, Neighbor edge, int c) const {
        const T delta = this->reference_value(r, edge, c) - q.values[c];
        if (this->scheme == 2) {
            const T rhoj = this->fields.density[edge.reference];
            return delta * this->fields.mass[edge.reference] * this->query_density[q.index] / (rhoj * rhoj);
        }
        return delta * r.weight();
    }
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T dx[D], w, gradient[D];
        const T distance = q.geometry.distance(r.data, dx);
        const T hi = q.support, hj = this->reference_supports[edge.reference];
        const T h = pairwise_support(hi, hj, this->support);
        if (mode == 1) {
            // Warp's Naive scheme uses the raw kernel Laplacian, even when
            // CRK or renormalization is supplied. grad-h still scales fj.
            const T factor = this->support == 15 ?
                (evaluate_kernel<T, D>(dx, hi, this->kernel).laplacian +
                 evaluate_kernel<T, D>(dx, hj, this->kernel).laplacian) / T(2) :
                 evaluate_kernel<T, D>(dx, h, this->kernel).laplacian;
            for (int c = 0; c < Fields; ++c) sum.values[c] += difference(q, r, edge, c) * factor;
            return;
        }
        // Every gradient-based variant has an exact zero self contribution,
        // including when the corrected gradient is nonzero at the origin.
        if (distance == T(0)) return;
        this->kernel_gradient(q, r, edge, dx, w, gradient);
        const T regularized = distance + T(1e-8) * h;
        T projection = T(0);
        for (int d = 0; d < D; ++d) projection += gradient[d] * dx[d] / regularized;
        if (mode == 3) {
            const T factor = projection / regularized;
            for (int base = 0; base < Fields; base += D) {
                T field_projection = T(0);
                for (int d = 0; d < D; ++d)
                    field_projection += difference(q, r, edge, base + d) * dx[d] / regularized;
                for (int d = 0; d < D; ++d)
                    sum.values[base + d] += (difference(q, r, edge, base + d) -
                        T(D + 2) * field_projection * dx[d] / regularized) * factor;
            }
        } else {
            const T denominator = mode == 4 ? distance + T(1e-12) * h : regularized;
            const T factor = -T(2) * projection / denominator;
            for (int c = 0; c < Fields; ++c) sum.values[c] += difference(q, r, edge, c) * factor;
        }
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        for (int c = 0; c < Fields; ++c) this->fields.output[q.index * Fields + c] = sum.values[c];
    }
};
} // namespace sph_native::neighborhood
