#pragma once
#include "execution/sphFields.h"

namespace sph_native::neighborhood {
// Derivative axes match the Warp ABI: dm1[component, derivative] and
// dm2[derivative, component, component]. Public rows count every admitted edge,
// including zero-weight edges, exactly as the Warp moment kernel does.
template <class T, int D, bool General = false> struct Moments : SphOperation<T, D, 0, General> {
    using Base = SphOperation<T, D, 0, General>;
    using typename Base::Query;
    using typename Base::Reference;
    static constexpr int components = 1 + 2 * D + 2 * D * D + D * D * D;
    using State = CountedSums<T, components>;
    static constexpr Weight weight = Weight::Volume;
    SPH_HD State identity(const Query&) const { return {}; }
    // Separate output arrays let the public adapter store canonical contiguous
    // tensors in one pass. The prototype may still use its packed output.
    T* moments[6] = {};
    template <class Record>
    SPH_HD void interact(const Query& q, const Record& r, Neighbor edge, State& sum) const {
        T dx[D], w, g;
        this->evaluate(q, r, edge, dx, w, g);
        T gradient[D]={};
        if constexpr(General) {
            T squared=0;for(int c=0;c<D;++c)squared+=dx[c]*dx[c];const T distance=scalar_sqrt(squared);
            const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=distance>eps?distance:eps;
            auto one=[&](T h,T scale) {
                if(h<=T(0)||distance>h)return;
                const T slope=warp_compat::gradient_shape_derivative<T,D>(distance/h,this->kernel,radial_shape<T,D>(distance/h,this->kernel).d1);
                const T factor=slope*normalization<T,D>(this->kernel)/power<T,D>(h,D+1);
                for(int c=0;c<D;++c)gradient[c]+=scale*((dx[c]/den)*factor);
            };
            const T hj=this->reference_supports[edge.reference];
            if(this->support==14||this->support==15){one(q.support,T(.5));one(hj,T(.5));}
            else one(pairwise_support(q.support,hj,this->support),T(1));
        } else for(int c=0;c<D;++c)gradient[c]=g*dx[c];
        const T volume = r.weight();
        constexpr int m1 = 1, m2 = m1 + D, dm0 = m2 + D * D;
        constexpr int dm1 = dm0 + D, dm2 = dm1 + D * D;
        sum.values[0] += volume * w;
        for (int a = 0; a < D; ++a) {
            sum.values[m1 + a] += dx[a] * (volume * w);
            sum.values[dm0 + a] += volume * gradient[a];
            for (int b = 0; b < D; ++b) {
                sum.values[m2 + a * D + b] += (dx[a] * dx[b]) * (volume * w);
                sum.values[dm1 + a * D + b] += volume * (dx[a] * gradient[b] + w * T(a == b));
                for (int c = 0; c < D; ++c)
                    sum.values[dm2 + c * D * D + a * D + b] += volume *
                        (dx[a] * dx[b] * gradient[c] + w *
                         (dx[a] * T(b == c) + T(a == c) * dx[b]));
            }
        }
        if constexpr (General) ++sum.count;
        else if (q.support > T(0) && q.geometry.distance(r.data, dx) < q.support) ++sum.count;
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output)
            for (int c = 0; c < components; ++c)
                this->fields.output[q.index * components + c] = sum.values[c];
        if (moments[0]) {
            constexpr int widths[6] = {1, D, D * D, D, D * D, D * D * D};
            int offset = 0;
            for (int tensor = 0; tensor < 6; ++tensor) {
                for (int c = 0; c < widths[tensor]; ++c)
                    moments[tensor][q.index * widths[tensor] + c] = sum.values[offset + c];
                offset += widths[tensor];
            }
        }
        if (this->fields.counts) this->fields.counts[q.index] = sum.count;
    }
};
} // namespace sph_native::neighborhood
