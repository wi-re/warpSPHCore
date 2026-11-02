#pragma once
#include "api.h"
#include "operators.h"
#include "execution/neighborhoodAdjoint.h"
#include "execution/atomic.h"
#include "kernels/evaluation.h"

namespace prototype {
using namespace sph_native;

template <class T> SPH_HD T optional(const void* pointer, int64_t at) {
    return pointer ? static_cast<const T*>(pointer)[at] : T(0);
}

template <class T, int D, int Fields> struct PairDerivative {
    T x[D] = {}, h = 0, mass = 0, density = 0;
    T query[Fields ? Fields : 1] = {}, reference[Fields ? Fields : 1] = {};
};

template <class T, int D> struct FinalizationPullback {
    const T* cache;
    const T* cotangent;
    T* accumulator_bars;
    SPH_HD void operator()(int64_t i) const {
        using Operation = Interpolate<T,D,1,true>;
        typename Operation::State state{};
        state.values[0] = cache[i*2]; state.values[1] = cache[i*2+1];
        const auto bars = Operation::finalize_vjp(state,cotangent+i);
        accumulator_bars[i*2] = bars.values[0];
        accumulator_bars[i*2+1] = bars.values[1];
    }
};

// First prototype: Wendland2, Gather, Difference gradient. The independent
// differentiation inputs are qx, rx, h_i, m_j, rho_j, f_i and f_j. Bounds and
// topology are fixed. Apparent-volume and corrected-CRK AD are not advertised.
template <class T, int D, int Fields, bool IsGradient, bool Normalize, bool IsDensity, bool FieldsOnly = false>
struct Pullback {
    SphFields<T, D, Fields> fields;
    Derivatives derivatives;
    const T* cache;
    static constexpr int QSlots = FieldsOnly ? Fields : D + 1 + Fields;
    static constexpr int RSlots = FieldsOnly ? Fields : D + 2 + Fields;
    using Query = PreparedQuery<T, D, Fields>;
    using Reference = ReferenceRecord<T, D, Fields>;
    using Loader = ReferenceLoader<T, D, Fields, IsDensity ? Weight::Mass : Weight::Volume>;

    SPH_HD auto query(int64_t i) const {
        return SphOperation<T, D, Fields>{fields}.prepare(i);
    }
    template <Owner Side> SPH_HD auto prepare(int64_t owner) const {
        if constexpr (Side == Owner::Query) return query(owner);
        else return Loader{fields}.load(owner);
    }
    template <Owner Side> SPH_HD auto identity() const {
        if constexpr (Side == Owner::Query) return Sums<T, QSlots>{};
        else return Sums<T, RSlots>{};
    }

    SPH_HD PairDerivative<T, D, Fields> pair(const Query& q, const Reference& r, int64_t j) const {
        PairDerivative<T, D, Fields> result;
        T dx[D];
        q.geometry.distance(r.data, dx);
        const auto kernel = evaluate_kernel<T, D>(dx, q.support, 0);
        const auto* bar = static_cast<const T*>(derivatives.cotangent);
        T bw = T(0), bg[D] = {}, bvolume = T(0);
        const T mass = fields.mass[j];
        const T density = IsDensity ? T(1) : fields.density[j];
        const T volume = IsDensity ? mass : mass / density;
        if constexpr (IsDensity) {
            bw = bar[q.index] * mass;
            result.mass = bar[q.index] * kernel.value;
        } else {
            for (int c = 0; c < Fields; ++c) {
                if constexpr (IsGradient) {
                    T coefficient_bar = T(0);
                    for (int d = 0; d < D; ++d) {
                        const T cot = bar[(q.index * Fields + c) * D + d];
                        coefficient_bar += cot * kernel.gradient[d];
                        if constexpr (!FieldsOnly)
                            bg[d] += cot * (r.value(c) - q.values[c]) * volume;
                    }
                    result.query[c] = -coefficient_bar * volume;
                    result.reference[c] = coefficient_bar * volume;
                    if constexpr (!FieldsOnly) bvolume += coefficient_bar * (r.value(c) - q.values[c]);
                } else {
                    T numerator_bar = bar[q.index * Fields + c], denominator_bar = T(0);
                    if constexpr (Normalize) {
                        const auto* final_bars = static_cast<const T*>(derivatives.final_bars);
                        numerator_bar = final_bars[q.index*(Fields+1)+c];
                        denominator_bar = final_bars[q.index*(Fields+1)+Fields];
                    }
                    result.reference[c] = numerator_bar * volume * kernel.value;
                    if constexpr (!FieldsOnly) {
                        const T weighted_bar = numerator_bar * r.value(c) + denominator_bar;
                        bw += weighted_bar * volume;
                        bvolume += weighted_bar * kernel.value;
                    }
                }
            }
            if constexpr (!FieldsOnly) {
                result.mass = bvolume / density;
                result.density = -bvolume * mass / (density * density);
            }
        }
        if constexpr (!FieldsOnly) {
            result.h = bw * kernel.dh;
            for (int d = 0; d < D; ++d) {
                result.x[d] = bw * kernel.gradient[d];
                result.h += bg[d] * kernel.gradient_dh[d];
                for (int c = 0; c < D; ++c)
                    result.x[d] += bg[c] * kernel.hessian[c * D + d];
            }
        }
        return result;
    }

    template <Owner Side, class Prepared, class State>
    SPH_HD void interact(int64_t owner, const Prepared& prepared, Neighbor neighbor, State& state) const {
        const int64_t i = Side == Owner::Query ? owner : neighbor.reference;
        const int64_t j = Side == Owner::Query ? neighbor.reference : owner;
        if ((fields.query_kinds && !query_allowed(fields.query_kinds[i], fields.direction)) ||
            (fields.reference_kinds && !reference_allowed(fields.reference_kinds[j], fields.direction))) return;
        const auto bars = [&]() {
            if constexpr (Side == Owner::Query) return pair(prepared, Loader{fields}.load(j), j);
            else return pair(query(i), prepared, j);
        }();
        if constexpr (FieldsOnly) {
            for (int c = 0; c < Fields; ++c)
                state.values[c] += Side == Owner::Query ? bars.query[c] : bars.reference[c];
        } else if constexpr (Side == Owner::Query) {
            for (int d = 0; d < D; ++d) state.values[d] += bars.x[d];
            state.values[D] += bars.h;
            for (int c = 0; c < Fields; ++c) state.values[D + 1 + c] += bars.query[c];
        } else {
            for (int d = 0; d < D; ++d) state.values[d] -= bars.x[d];
            state.values[D] += bars.mass;
            state.values[D + 1] += bars.density;
            for (int c = 0; c < Fields; ++c) state.values[D + 2 + c] += bars.reference[c];
        }
    }

    template <Owner Side, class State> SPH_HD void finish(int64_t owner, const State& state) const {
        if constexpr (FieldsOnly) {
            void* pointer = Side == Owner::Query ? derivatives.gqvalue : derivatives.grvalue;
            if (pointer)
                for (int c = 0; c < Fields; ++c) static_cast<T*>(pointer)[owner * Fields + c] = state.values[c];
        } else if constexpr (Side == Owner::Query) {
            if (derivatives.gx)
                for (int d = 0; d < D; ++d) static_cast<T*>(derivatives.gx)[owner * D + d] = state.values[d];
            if (derivatives.gh) static_cast<T*>(derivatives.gh)[owner] = state.values[D];
            if (derivatives.gqvalue)
                for (int c = 0; c < Fields; ++c) static_cast<T*>(derivatives.gqvalue)[owner * Fields + c] = state.values[D + 1 + c];
        } else {
            if (derivatives.grx)
                for (int d = 0; d < D; ++d) static_cast<T*>(derivatives.grx)[owner * D + d] = state.values[d];
            if (derivatives.gm) static_cast<T*>(derivatives.gm)[owner] = state.values[D];
            if (derivatives.grho) static_cast<T*>(derivatives.grho)[owner] = state.values[D + 1];
            if (derivatives.grvalue)
                for (int c = 0; c < Fields; ++c) static_cast<T*>(derivatives.grvalue)[owner * Fields + c] = state.values[D + 2 + c];
        }
    }
};

template <class T, int D, int Fields, bool IsGradient, bool Normalize, bool IsDensity, bool FieldsOnly = false>
struct Tangent : SphOperation<T, D, Fields> {
    using Base = SphOperation<T, D, Fields>;
    using typename Base::Query;
    using typename Base::Reference;
    static constexpr int Outputs = IsDensity ? 1 : IsGradient ? Fields * D : Fields;
    static constexpr int Primal = Fields + int(Normalize);
    using State = Sums<T, Outputs + (Normalize ? 3 : 0)>;
    static constexpr Weight weight = IsDensity ? Weight::Mass : Weight::Volume;
    Derivatives derivatives;

    SPH_HD State identity(const Query&) const { return {}; }
    SPH_HD void interact(const Query& q, const Reference& r, Neighbor neighbor, State& sum) const {
        const auto i = q.index, j = neighbor.reference;
        const auto& f = this->fields;
        T dx[D], tangent_dx[D], dw = T(0), gradient_tangent[D] = {};
        q.geometry.distance(r.data, dx);
        const auto kernel = evaluate_kernel<T, D>(dx, q.support, 0);
        const T dh = optional<T>(derivatives.th, i);
        if constexpr (!FieldsOnly) {
            dw = kernel.dh * dh;
            for (int d = 0; d < D; ++d) {
                tangent_dx[d] = optional<T>(derivatives.tx, i * D + d) - optional<T>(derivatives.trx, j * D + d);
                dw += kernel.gradient[d] * tangent_dx[d];
            }
            for (int d = 0; d < D; ++d) {
                gradient_tangent[d] = kernel.gradient_dh[d] * dh;
                for (int c = 0; c < D; ++c) gradient_tangent[d] += kernel.hessian[d * D + c] * tangent_dx[c];
            }
        }
        const T dm = optional<T>(derivatives.tm, j);
        if constexpr (IsDensity) sum.values[0] += dm * kernel.value + f.mass[j] * dw;
        else {
            const T rho = f.density[j], volume = f.mass[j] / rho;
            T dv = T(0);
            if constexpr (!FieldsOnly) dv = dm / rho - f.mass[j] * optional<T>(derivatives.trho, j) / (rho * rho);
            for (int c = 0; c < Fields; ++c) {
                const T drvalue = optional<T>(derivatives.trvalue, j * Fields + c);
                if constexpr (IsGradient) {
                    const T difference = r.value(c) - q.values[c];
                    const T ddifference = drvalue - optional<T>(derivatives.tqvalue, i * Fields + c);
                    for (int d = 0; d < D; ++d)
                        sum.values[c * D + d] += (ddifference * volume + difference * dv) * kernel.gradient[d]
                            + difference * volume * gradient_tangent[d];
                } else sum.values[c] += drvalue * volume * kernel.value + r.value(c) * (dv * kernel.value + volume * dw);
            }
            if constexpr (Normalize) {
                sum.values[2] += volume * kernel.value;
                if constexpr (!FieldsOnly) {
                    sum.values[1] += r.value(0) * volume * kernel.value;
                    sum.values[3] += dv * kernel.value + volume * dw;
                }
            }
        }
    }
    SPH_HD void finish(const Query& q, const State& sum) const {
        if (!this->fields.output) return;
        for (int c = 0; c < Outputs; ++c) {
            T value = sum.values[c];
            if constexpr (Normalize) {
                const T numerator = sum.values[1], denominator = sum.values[2];
                if constexpr (FieldsOnly) value = denominator != T(0) ? value / denominator : T(0);
                else value = denominator != T(0) ?
                    value / denominator - numerator * sum.values[3] / (denominator * denominator) : T(0);
            }
            this->fields.output[q.index * Outputs + c] = value;
        }
    }
};

// Matched arithmetic control for the existing scatter design. Ownership and
// initialization differ; pair derivatives are exactly the same as owned VJP.
template <class T, int D, int Fields, class PullbackType, class Traversal>
struct AtomicAdjointRow {
    PullbackType pullback;
    Traversal traversal;

    struct Visit {
        const PullbackType& pb;
        const PreparedQuery<T, D, Fields>& query;
        int64_t i;
        SPH_HD void operator()(Neighbor neighbor) const {
            const int64_t j = neighbor.reference;
            if ((pb.fields.query_kinds && !query_allowed(pb.fields.query_kinds[i], pb.fields.direction)) ||
                (pb.fields.reference_kinds && !reference_allowed(pb.fields.reference_kinds[j], pb.fields.direction))) return;
            const auto reference = typename PullbackType::Loader{pb.fields}.load(j);
            const auto bars = pb.pair(query, reference, j);
            const auto& out = pb.derivatives;
            for (int d = 0; d < D; ++d) {
                if (out.gx) add_atomic(static_cast<T*>(out.gx) + i * D + d, bars.x[d]);
                if (out.grx) add_atomic(static_cast<T*>(out.grx) + j * D + d, -bars.x[d]);
            }
            if (out.gh) add_atomic(static_cast<T*>(out.gh) + i, bars.h);
            if (out.gm) add_atomic(static_cast<T*>(out.gm) + j, bars.mass);
            if (out.grho) add_atomic(static_cast<T*>(out.grho) + j, bars.density);
            for (int c = 0; c < Fields; ++c) {
                if (out.gqvalue) add_atomic(static_cast<T*>(out.gqvalue) + i * Fields + c, bars.query[c]);
                if (out.grvalue) add_atomic(static_cast<T*>(out.grvalue) + j * Fields + c, bars.reference[c]);
            }
        }
    };
    template <class Reduction>
    SPH_HD void run(int64_t i, int lane, int lanes, const Reduction&) const {
        const auto query = pullback.query(i);
        traversal.each(i, lane, lanes, Visit{pullback, query, i});
    }
    SPH_HD void operator()(int64_t i) const { run(i, 0, 1, LocalReduction{}); }
};
} // namespace prototype
