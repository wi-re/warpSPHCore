#pragma once
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/library.h>
#include <vector>
#include "dataTypes/slots.h"
namespace sph_native {
namespace stable = torch::stable;
using Tensor = stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using Tensors = std::vector<Tensor>;
} // namespace sph_native
