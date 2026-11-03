#pragma once
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/library.h>
#include <vector>
#include <optional>
#include "dataTypes/slots.h"
namespace sph_native {
namespace stable = torch::stable;
using Tensor = stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using Tensors = std::vector<Tensor>;
using OptionalTensors = std::vector<std::optional<Tensor>>;
inline const Tensor* optional_tensor(const OptionalTensors& values, int slot) {
    return values[slot] ? &*values[slot] : nullptr;
}
} // namespace sph_native
