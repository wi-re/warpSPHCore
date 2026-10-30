#pragma once
#include "torch/types.h"
namespace sph_native {
void tensor(const Tensor &, const Tensor &, ScalarType, const char *);
void vector(const Tensor &, const Tensor &, ScalarType, int64_t, const char *);
void finite(const Tensor &, const char *);
void geometry(const Tensors &, int);
Tensors prepare_operator(const Tensors &, int support, int direction, bool gradient);
Tensors prepare_field_operator(const Tensors &, int, int, int, int, int, bool);
} // namespace sph_native
