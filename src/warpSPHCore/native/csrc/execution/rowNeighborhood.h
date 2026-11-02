#pragma once

#include "dataTypes/state.h"
#include "execution/neighborhood.h"
#include "execution/sphFields.h"

namespace sph_native::neighborhood {

// Adapt the established borrowed tensor view once per launch. Neither this
// adapter nor the numerical operation depends on Torch's tensor implementation.
template <class T, int D, int Fields>
SphFields<T, D, Fields> sph_fields(const RowArgs<T, D>& a) {
    SphFields<T, D, Fields> fields{};
    fields.query_positions = a.qx;
    fields.reference_positions = a.rx;
    fields.supports = a.qh;
    fields.mass = a.mass;
    fields.density = a.density;
    fields.query_values = a.qvalue;
    fields.reference_values = a.rvalue;
    fields.lower = a.lo;
    fields.upper = a.hi;
    fields.periodic = a.periodic;
    fields.direction = a.direction;
    fields.query_kinds = a.direction == 0 ? nullptr : a.qkind;
    fields.reference_kinds = a.direction == 0 ? nullptr : a.rkind;
    fields.output = a.output;
    return fields;
}

template <class Index, class T, int D>
Csr<Index, IndexView> csr_view(const RowArgs<T, D>& a) {
    return {a.offsets, a.counts, static_cast<const Index*>(a.neighbors.data)};
}

template <class Operation, class Loader, class T, int D, class Launch>
void launch_serial_csr(int64_t n, Operation operation, Loader loader,
                       const RowArgs<T, D>& a, const Launch& launch) {
    if (a.neighbors.wide)
        launch(n, Row<Operation, Csr<int64_t, IndexView>, Loader>{
            operation, csr_view<int64_t>(a), loader});
    else
        launch(n, Row<Operation, Csr<int32_t, IndexView>, Loader>{
            operation, csr_view<int32_t>(a), loader});
}

// The policy declares its field requirements. CPU loaders read borrowed arrays
// directly; only GPU execution prepares packed records for reuse over edges.
template <template <class, int, int> class Policy, Weight Kind> struct SerialSphRows {
    template <class T, int D, int Fields, class Launch>
    void fields(int64_t n, RowArgs<T, D> a, const Launch& launch) const {
        using Operation = Policy<T, D, Fields>;
        constexpr int stored = Operation::field_count;
        auto input = sph_fields<T, D, stored>(a);
        if constexpr (!Operation::uses_query_values) input.query_values = nullptr;
        Operation operation{};
        operation.fields = input;
        launch_serial_csr(n, operation, BorrowedReferenceLoader<T, D, stored, Kind>{input}, a, launch);
    }

    template <class T, int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch& launch) const {
        if (a.fields == 1) fields<T, D, 1>(n, a, launch);
        else if (a.fields == D) fields<T, D, D>(n, a, launch);
        else fields<T, D, D * D>(n, a, launch);
    }
};

} // namespace sph_native::neighborhood
