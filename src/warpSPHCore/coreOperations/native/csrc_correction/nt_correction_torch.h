#pragma once
#include "torch/validation.h"
#include <optional>

namespace sph_native {
using OptionalTensors = std::vector<std::optional<Tensor>>;
enum CorrectionSlot { QUERY_VOLUME, REFERENCE_VOLUME, CRK_A, CRK_B,
                      CRK_GRAD_A, CRK_GRAD_B, REFERENCE_OMEGA, RENORMALIZATION,
                      CORRECTION_SLOTS };

inline const Tensor* optional_tensor(const OptionalTensors& values, int slot) {
    return values[slot] ? &*values[slot] : nullptr;
}

// These checks concern tensor metadata only. In particular, no adjacency or
// coefficient contents are scanned and no device buffers are read on the CPU.
inline Tensors prepare_corrected_operator(const OptionalTensors& inputs,
        const OptionalTensors& corrections, int support, int direction,
        int operation, int scheme) {
    STD_TORCH_CHECK(inputs.size() == OUTPUT, "invalid corrected operator tensor layout");
    STD_TORCH_CHECK(corrections.size() == CORRECTION_SLOTS,
                    "invalid native correction tensor layout");
    Tensors borrowed;
    borrowed.reserve(inputs.size());
    for (const auto& value : inputs) borrowed.push_back(value ? *value : Tensor{});
    auto t = prepare_pair_geometry(borrowed, support, direction);
    const auto& x = t[QX];
    const auto n = x.size(0), nr = t[RX].size(0), dimension = x.size(1);
    for (auto slot : {QUERY_VOLUME, REFERENCE_VOLUME, CRK_A, REFERENCE_OMEGA}) {
        if (const auto* value = optional_tensor(corrections, slot))
            vector(*value, x, x.scalar_type(),
                slot == REFERENCE_VOLUME || slot == REFERENCE_OMEGA ? nr : n,
                "native correction scalar");
    }
    const bool crk = bool(corrections[CRK_A]);
    STD_TORCH_CHECK(crk == bool(corrections[CRK_B]), "CRK requires both A and B");
    auto shaped = [&](int slot, int rank) {
        const auto* value = optional_tensor(corrections, slot);
        STD_TORCH_CHECK(value, "missing CRK gradient coefficients");
        tensor(*value, x, x.scalar_type(), "native correction coefficients");
        STD_TORCH_CHECK(value->dim() == rank + 1 && value->size(0) == n,
                        "correction coefficients have an incompatible shape");
        for (int d = 1; d <= rank; ++d)
            STD_TORCH_CHECK(value->size(d) == dimension,
                            "correction axes must match spatial dimension");
    };
    if (crk) {
        shaped(CRK_B, 1);
        if (operation == 2) { shaped(CRK_GRAD_A, 1); shaped(CRK_GRAD_B, 2); }
    } else {
        STD_TORCH_CHECK(!corrections[CRK_GRAD_A] && !corrections[CRK_GRAD_B],
                        "CRK gradient coefficients require A and B");
    }
    if (corrections[RENORMALIZATION]) shaped(RENORMALIZATION, 2);
    STD_TORCH_CHECK(operation == 2 ||
        (!corrections[REFERENCE_OMEGA] && !corrections[RENORMALIZATION]),
        "grad-h and renormalization are gradient corrections");
    // Symmetric weighting uses masses/densities even with explicit volumes,
    // matching the reference implementation. Other schemes can omit them.
    if (!corrections[REFERENCE_VOLUME] || (operation == 2 && scheme == 2)) {
        vector(t[MASS], x, x.scalar_type(), nr, "reference masses");
        vector(t[DENSITY], x, x.scalar_type(), nr, "reference densities");
    }
    return prepare_field_output(std::move(t), operation, scheme, 2, false);
}
} // namespace sph_native
