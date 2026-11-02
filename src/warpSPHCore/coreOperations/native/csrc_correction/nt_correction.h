#pragma once
#include "execution/sphFields.h"
#include "crk/native/csrc_correction/nt_correction.h"

namespace sph_native::neighborhood {
// Only corrected operations carry these pointers. Ordinary operation/loader
// layouts and their tuned dispatch remain untouched.
template <class T> struct CorrectionFields {
    crk::Fields<T> crk;
    const T* reference_omega = nullptr;
    const T* renormalization = nullptr;
    const T* query_omega = nullptr;
};
// Shared corrected-kernel preparation. The operator chooses field weighting,
// tensor contraction and finalization; traversal and reference loading stay
// independent of that numerical policy.
template <class T, int D, class Base, bool ScaleQueryOmega = false> struct CorrectedKernel : Base {
    struct Query : Base::Query {
        crk::Coefficients<T, D, true> correction;
        T renormalization[D * D] = {};
    };
    CorrectionFields<T> corrections;

    SPH_HD Query prepare(int64_t i) const {
        Query q{Base::prepare(i), crk::prepare<T, D, true>(corrections.crk, i)};
        if (corrections.renormalization)
            for (int c = 0; c < D * D; ++c)
                q.renormalization[c] = corrections.renormalization[i * D * D + c];
        if constexpr (ScaleQueryOmega) {
            if (corrections.query_omega)
                for (int c = 0; c < Base::field_count; ++c)
                    q.values[c] /= corrections.query_omega[i];
        }
        return q;
    }
    template <class Record>
    SPH_HD void kernel_gradient(const Query& q, const Record& r, Neighbor edge,
            T (&dx)[D], T& w, T (&gradient)[D]) const {
        T radial_gradient;
        this->evaluate(q, r, edge, dx, w, radial_gradient);
        if (corrections.crk.A) crk::gradient(q.correction, dx, w, radial_gradient, gradient);
        else for (int d = 0; d < D; ++d) gradient[d] = radial_gradient * dx[d];
        if (corrections.renormalization) {
            T transformed[D] = {};
            for (int d = 0; d < D; ++d)
                for (int c = 0; c < D; ++c)
                    transformed[d] += q.renormalization[d * D + c] * gradient[c];
            for (int d = 0; d < D; ++d) gradient[d] = transformed[d];
        }
    }
    template <class Record>
    SPH_HD T reference_value(const Record& r, Neighbor edge, int c) const {
        const T fj = r.value(c);
        return corrections.reference_omega ? fj / corrections.reference_omega[edge.reference] : fj;
    }
};
} // namespace sph_native::neighborhood
