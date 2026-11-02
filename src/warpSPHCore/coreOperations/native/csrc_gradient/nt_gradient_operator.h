#pragma once
#include "execution/sphFields.h"
#include "coreOperations/native/csrc_gradient/nt_gradient_forward.h"

namespace sph_native::neighborhood {
template <class T, int D, int Fields = 1>
struct Gradient : SphOperation<T, D, Fields> {
    using Base = SphOperation<T, D, Fields>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = Sums<T, Fields * D>;
    static constexpr Weight weight = Weight::Volume;
    SPH_HD State identity(const Query&) const { return {}; }
    SPH_HD void interact(const Query& q, const Reference& r, Neighbor neighbor, State& sum) const {
        T dx[D], w, g;
        const T distance = q.geometry.distance(r.data, dx);
        q.kernel.evaluate(distance, w, g);
        // Reuse the existing operator's actual pair arithmetic. The adapter
        // supplies a prepared source record and a cached query field, replacing
        // tensor gathers without changing the established Difference formula.
        PackedFieldArgs<T, D> args{};
        args.reference = r.data;
        args.query_field = q.values;
        args.fields = Fields;
        T contribution[D * D * D] = {};
        GradientPair<T, D>::accumulate(args, q.index, neighbor.reference, dx,
                                       distance, w, g, contribution);
        for (int c = 0; c < Fields * D; ++c) sum.values[c] += contribution[c];
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (this->fields.output)
            for (int c = 0; c < Fields * D; ++c)
                this->fields.output[q.index * Fields * D + c] = sum.values[c];
    }
};
} // namespace sph_native::neighborhood
