#pragma once
#include "execution/dispatch.h"
#include "execution/neighborhoodAdjoint.h"
#include "coreOperations/native/csrc_correction/nt_correction_rows.h"
#include "coreOperations/native/csrc_interpolate/nt_interpolate_operator.h"
#include "coreOperations/native/csrc_gradient/nt_gradient_operator.h"
#include "coreOperations/native/csrc_divergence/nt_divergence_operator.h"
#include "coreOperations/native/csrc_curl/nt_curl_operator.h"
#include "coreOperations/native/csrc_laplacian/nt_laplacian_operator.h"

namespace sph_native::neighborhood {
template <int Width, class PortableRow> struct CooperativeRow;
// The field derivative never reads primal fields or constructs geometric
// Hessians. Nullable tangent pointers are exact zero inputs, not dummy arrays.
template <class T> struct FieldRecord {
    const T* data;
    const T* values;
    T volume;
    SPH_HD T weight() const { return volume; }
    SPH_HD T value(int c) const { return values ? values[c] : T(0); }
};
template <class T, int D, int Fields> struct FieldLoader {
    SphFields<T, D, Fields> fields;
    SPH_HD bool allowed(int64_t j) const {
        return !fields.reference_kinds || reference_allowed(fields.reference_kinds[j], fields.direction);
    }
    SPH_HD FieldRecord<T> load(int64_t j) const {
        return {fields.reference_positions + j * D,
                fields.reference_values ? fields.reference_values + j * Fields : nullptr,
                fields.volumes ? fields.volumes[j] : fields.mass[j] / fields.density[j]};
    }
};

template <class T, int D, int Fields, class Operation> struct FieldPullback {
    Operation operation;
    FieldLoader<T, D, Fields> loader;
    const T* cotangent;
    int outputs;
    T *query, *reference;
    template <Owner Side> SPH_HD auto prepare(int64_t owner) const {
        if constexpr (Side == Owner::Query) return operation.prepare(owner);
        else return loader.load(owner);
    }
    template <Owner Side> SPH_HD Sums<T, Fields> identity() const { return {}; }
    template <Owner Side, class Prepared>
    SPH_HD void interact(int64_t owner, const Prepared& prepared, Neighbor neighbor, Sums<T, Fields>& sum) const {
        // CSC executes the original i,j interaction, including h_i/h_j and
        // query-side corrections. It never swaps the physical argument roles.
        const int64_t i = Side == Owner::Query ? owner : neighbor.reference;
        const int64_t j = Side == Owner::Query ? neighbor.reference : owner;
        if (!operation.allowed(i) || !loader.allowed(j)) return;
        const auto q = [&]() {
            if constexpr (Side == Owner::Query) return prepared;
            else return operation.prepare(i);
        }();
        const auto r = [&]() {
            if constexpr (Side == Owner::Query) return loader.load(j);
            else return prepared;
        }();
        const Neighbor original{j, -1};
        T contraction[Fields] = {};
        operation.field_contract(q, r, original, cotangent + i * outputs, contraction);
        const auto scales = operation.field_scales(q, r, original);
        const T scale = Side == Owner::Query ? scales.query : scales.reference;
        for (int c = 0; c < Fields; ++c) sum.values[c] += scale * contraction[c];
    }
    template <Owner Side> SPH_HD void finish(int64_t owner, const Sums<T, Fields>& sum) const {
        T* result = Side == Owner::Query ? query : reference;
        for (int c = 0; c < Fields; ++c) result[owner * Fields + c] = sum.values[c];
    }
};
struct FieldBuffers {
    Tensor query, reference, cotangent;
    Tensors transpose;
};

template <bool Device, bool Pullback> struct FieldExecutor {
    static constexpr bool device = Device;
    const FieldBuffers& buffers;

    template <Owner Side, class Pull, class Launch>
    void owned(int64_t n, Pull pullback, IndexView offsets, const int32_t* counts,
               IndexView neighbors, const Launch& launch) const {
        auto typed = [&](auto* ids) {
            using Index = std::remove_cv_t<std::remove_pointer_t<decltype(ids)>>;
            using Traversal = Csr<Index, IndexView>;
            using Row = OwnedAdjointRow<Side, Pull, Traversal>;
            const Row row{pullback, {offsets, counts, ids}};
            if constexpr (Device) launch(n, CooperativeRow<Launch::width, Row>{row});
            else launch(n, row);
        };
        if (neighbors.wide) typed(static_cast<const int64_t*>(neighbors.data));
        else typed(static_cast<const int32_t*>(neighbors.data));
    }
    template <class Operation, class Ignored, class T, int D, class Launch>
    void operator()(int64_t n, Operation op, Ignored, const RowArgs<T, D>& a, const Launch& launch) const {
        constexpr int fields = Operation::field_count;
        if constexpr (!Pullback) {
            op.fields.query_values = ptr<T>(buffers.query);
            op.fields.reference_values = ptr<T>(buffers.reference);
            const FieldLoader<T, D, fields> loader{op.fields};
            if constexpr (Device) launch_cooperative_csr(n, op, loader, a, launch);
            else launch_serial_csr(n, op, loader, a, launch);
        } else {
            op.fields.query_values = op.fields.reference_values = nullptr;
            op.fields.output = nullptr;
            const FieldPullback<T, D, fields, Operation> pullback{op, {op.fields},
                buffers.cotangent.const_data_ptr<T>(), a.outputs,
                buffers.query.defined() ? buffers.query.mutable_data_ptr<T>() : nullptr,
                buffers.reference.defined() ? buffers.reference.mutable_data_ptr<T>() : nullptr};
            if (buffers.query.defined())
                owned<Owner::Query>(n, pullback, a.offsets, a.counts, a.neighbors, launch);
            if (buffers.reference.defined()) {
                const auto& csc = buffers.transpose;
                owned<Owner::Reference>(a.references, pullback, index_view(csc[0]),
                    csc[1].const_data_ptr<int32_t>(), index_view(csc[2]), launch);
            }
        }
    }
};
} // namespace sph_native::neighborhood
