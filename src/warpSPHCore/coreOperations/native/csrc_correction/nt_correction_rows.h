#pragma once
#include "execution/rowNeighborhood.h"
#include "coreOperations/native/csrc_correction/nt_correction_torch.h"
#include "coreOperations/native/csrc_correction/nt_correction.h"

namespace sph_native::neighborhood {
struct SerialCorrectionExecutor {
    static constexpr bool device = false;
    template <class Operation, class Loader, class T, int D, class Launch>
    void operator()(int64_t n, Operation op, Loader loader, const RowArgs<T, D>& a,
                    const Launch& launch) const {
        launch_serial_csr(n, op, loader, a, launch);
    }
};

// Adapt nullable Torch handles once per launch. Optional coefficients become
// nullptr; reference volumes enter the existing reference loader directly.
struct NoCorrectionOptions {
    template <class Operation> void operator()(Operation&) const {}
};
template <template <class, int, int> class Policy, class Executor,
          class Options = NoCorrectionOptions>
struct CorrectedRows {
    const OptionalTensors& corrections;
    Options options{};
    Executor executor{};

    template <class T> const T* pointer(int slot) const {
        const auto* value = optional_tensor(corrections, slot);
        return value ? value->const_data_ptr<T>() : nullptr;
    }

    template <class T, int D, int Fields, class Launch>
    void fields(int64_t n, RowArgs<T, D> a, const Launch& launch) const {
        using Operation = Policy<T, D, Fields>;
        constexpr int stored = Operation::field_count;
        auto input = sph_fields<T, D, stored>(a);
        input.volumes = pointer<T>(REFERENCE_VOLUME);
        if constexpr (!Operation::uses_query_values) input.query_values = nullptr;
        Operation op{};
        op.fields = input;
        op.configure(a);
        op.corrections = {{pointer<T>(CRK_A), pointer<T>(CRK_B),
                           pointer<T>(CRK_GRAD_A), pointer<T>(CRK_GRAD_B)},
                          pointer<T>(REFERENCE_OMEGA), pointer<T>(RENORMALIZATION), pointer<T>(QUERY_OMEGA)};
        options(op);
        if constexpr (Executor::device) {
            executor(n, op, ReferenceLoader<T, D, stored, Weight::Volume>{input}, a, launch);
        } else {
            executor(n, op, BorrowedReferenceLoader<T, D, stored, Weight::Volume>{input}, a, launch);
        }
    }

    template <class T, int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch& launch) const {
        if (a.fields == 1) fields<T, D, 1>(n, a, launch);
        else if (a.fields == D) fields<T, D, D>(n, a, launch);
        else fields<T, D, D * D>(n, a, launch);
    }
};
} // namespace sph_native::neighborhood
