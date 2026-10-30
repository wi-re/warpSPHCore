#pragma once
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/library.h>
#include <vector>
namespace sph_native {
namespace stable = torch::stable;
using Tensor = stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
enum Slot { QX, RX, QH, RH, MASS, QKIND, RKIND, LO, HI, PERIODIC,
            OFFSETS, COUNTS, NEIGHBORS, QVALUE, RVALUE, DENSITY, OUTPUT, SLOTS };
using Tensors = std::vector<Tensor>;
} // namespace sph_native
