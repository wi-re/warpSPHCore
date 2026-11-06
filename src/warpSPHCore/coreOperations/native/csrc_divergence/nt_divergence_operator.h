#pragma once
#include "coreOperations/native/csrc_gradient/nt_gradient_operator.h"

namespace sph_native::neighborhood {
template <class T, int D, int Fields> struct CorrectedDivergence
    : CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>, true> {
    using Base = CorrectedKernel<T, D, GeneralGradientPolicy<T, D, Fields>, true>;
    using typename Base::Query;
    static constexpr int components = Fields < D ? 1 : Fields / D;
    using State = Sums<T, components>;
    bool dot_mode = false, consistent = false;
    template <class Args> void configure(const Args& a) {
        Base::configure(a);
        dot_mode = a.dot_mode;
    }
    SPH_HD State identity(const Query&) const { return {}; }
    template <class Record>
    SPH_HD FieldScales<T> field_scales(const Query& q, const Record& r, Neighbor edge) const {
        if(consistent && this->scheme!=2) {
            const T v=consistent_volume(q,edge);
            return this->scale_fields(q.index,edge,FieldScales<T>{this->scheme==1 ? T(0):this->scheme==3 ? -v:v,v});
        }
        return Base::field_scales(q,r,edge);
    }
    SPH_HD T consistent_volume(const Query& q,Neighbor edge) const {
        const auto j=edge.reference;
        return this->fields.volumes ? this->fields.volumes[j]*this->fields.density[j]/this->query_density[q.index] :
            this->fields.mass[j]/this->query_density[q.index];
    }
    template <class Record>
    SPH_HD void field_contract(const Query& q, const Record& r, Neighbor edge,
                              const T* cotangent, T* result) const {
        T dx[D], w, gradient[D];
        this->kernel_gradient(q, r, edge, dx, w, gradient);
        for (int c = 0; c < components; ++c)
            for (int d = 0; d < D; ++d)
                result[dot_mode ? d * components + c : c * D + d] += cotangent[c] * gradient[d];
    }

    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T dx[D], w, gradient[D];
        this->kernel_gradient(q, r, edge, dx, w, gradient);
        for (int c = 0; c < components; ++c)
            for (int d = 0; d < D; ++d) {
                const int field = dot_mode ? d * components + c : c * D + d;
                T weighted = this->weighted_value(q, r, edge, field, this->reference_value(r, edge, field));
                // Symmetric's mass/density formula is unchanged by the
                // consistency option, just as in the reference operator.
                if (consistent && this->scheme != 2) {
                    const T fi=q.values[field],fj=this->reference_value(r,edge,field);
                    weighted=(this->scheme==1 ? fj:this->scheme==3 ? fj-fi:fj+fi)*consistent_volume(q,edge);
                }
                sum.values[c] += weighted * gradient[d];
            }
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        for (int c = 0; c < components; ++c) this->fields.output[q.index * components + c] = sum.values[c];
    }
};
struct DivergenceCorrectionOptions {
    bool consistent;
    template <class Operation> void operator()(Operation& op) const { op.consistent = consistent; }
};
} // namespace sph_native::neighborhood
