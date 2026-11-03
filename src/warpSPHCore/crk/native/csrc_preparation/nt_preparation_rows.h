#pragma once
#include "execution/rowNeighborhood.h"
#include "crk/native/csrc_preparation/nt_preparation_torch.h"
#include "crk/native/csrc_moments/nt_moments_operator.h"
#include "crk/native/csrc_density/nt_density_operator.h"

namespace sph_native::crk {
// This adapter only borrows tensors. Physics stays in per-function policies;
// alternate neighborhood executors can consume the same policies and loaders.
template <bool MomentPass, class Executor> struct PreparationRows {
    const PreparationTensors& tensors;
    template <class T, int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch& launch) const {
        auto fields = neighborhood::sph_fields<T, D, 0>(a);
        fields.volumes = tensors.volumes.const_data_ptr<T>();
        using Operation = std::conditional_t<MomentPass,
            neighborhood::Moments<T, D, true>, neighborhood::ConsistencyDensity<T, D>>;
        Operation op{};
        if constexpr (MomentPass) {
            fields.counts = tensors.counts.mutable_data_ptr<int32_t>();
            for (int k = 0; k < 6; ++k) op.moments[k] = tensors.moments[k].mutable_data_ptr<T>();
        } else {
            op.coefficients = {tensors.A.const_data_ptr<T>(), tensors.B.const_data_ptr<T>()};
        }
        op.fields = fields;
        op.configure(a);
        if constexpr (Executor::device)
            Executor{}(n, op, neighborhood::ReferenceLoader<T, D, 0, neighborhood::Weight::Volume>{fields}, a, launch);
        else
            Executor{}(n, op, neighborhood::BorrowedReferenceLoader<T, D, 0, neighborhood::Weight::Volume>{fields}, a, launch);
    }
};
struct SerialPreparationExecutor {
    static constexpr bool device = false;
    template <class Operation, class Loader, class T, int D, class Launch>
    void operator()(int64_t n, Operation op, Loader loader, const RowArgs<T, D>& a,
                    const Launch& launch) const {
        neighborhood::launch_serial_csr(n, op, loader, a, launch);
    }
};
} // namespace sph_native::crk
