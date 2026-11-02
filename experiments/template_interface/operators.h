#pragma once
#include "coreOperations/native/csrc_density/nt_density_operator.h"
#include "coreOperations/native/csrc_interpolate/nt_interpolate_operator.h"
#include "coreOperations/native/csrc_gradient/nt_gradient_operator.h"
#include "coreOperations/native/csrc_covariance/nt_covariance_operator.h"
#include "crk/native/csrc_moments/nt_moments_operator.h"

namespace prototype {
using namespace sph_native::neighborhood;

// A small winner reducer tests non-additive combination and optional outputs.
// Ties deliberately use the smallest reference ID across all traversal formats.
template <class T, int D> struct MaxField : SphOperation<T, D, 1> {
    using Base = SphOperation<T, D, 1>;
    using typename Base::Query;
    using typename Base::Reference;
    using State = Maximum<T>;
    static constexpr Weight weight = Weight::None;
    SPH_HD State identity(const Query&) const { return {}; }
    SPH_HD void interact(const Query& q, const Reference& r, Neighbor neighbor, State& state) const {
        T dx[D];
        const T distance = q.geometry.distance(r.data, dx);
        if (q.support > T(0) && distance < q.support)
            state.consider(r.value(0), int32_t(neighbor.reference));
    }
    SPH_HD void finish(const Query& q, const State& state) const {
        if (this->fields.output) this->fields.output[q.index] = state.index >= 0 ? state.value : T(0);
        if (this->fields.winners) this->fields.winners[q.index] = state.index;
    }
};
} // namespace prototype
