#pragma once
#include "torch/validation.h"
#include "kernels/value.h"

namespace sph_native::crk {
// Metadata checks only. Neither CSR contents nor particle values are scanned.
inline Tensors prepare(const OptionalTensors& inputs, int support, int direction, int kernel) {
    STD_TORCH_CHECK(valid_kernel(kernel), "Unknown native kernel");
    STD_TORCH_CHECK(inputs.size() == OUTPUT, "invalid CRK preparation tensor layout");
    Tensors borrowed;
    borrowed.reserve(inputs.size());
    for (const auto& value : inputs) borrowed.push_back(value ? *value : Tensor{});
    return prepare_pair_geometry(borrowed, support, direction);
}
inline void volume(const Tensor& value, const Tensors& t) {
    vector(value, t[QX], t[QX].scalar_type(), t[RX].size(0), "reference volumes");
}
struct PreparationTensors {
    Tensor volumes, A, B;
    Tensors moments;
    Tensor counts;
};
} // namespace sph_native::crk
