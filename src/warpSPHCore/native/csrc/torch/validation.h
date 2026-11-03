#pragma once
#include "torch/types.h"
namespace sph_native {
void tensor(const Tensor &, const Tensor &, ScalarType, const char *);
void vector(const Tensor &, const Tensor &, ScalarType, int64_t, const char *);
void geometry(const Tensors &, int);
Tensors prepare_operator(const Tensors &, int support, int direction);
Tensors prepare_pair_geometry(const Tensors &, int support, int direction);
Tensors prepare_field_output(Tensors, int operation, int scheme, int lap, bool dot, const Tensor* output = nullptr);
Tensors prepare_field_operator(const Tensors &, int, int, int, int, int, bool);
} // namespace sph_native
