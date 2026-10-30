#pragma once
#include "torch/types.h"
namespace sph_native {
template <class T, int D> struct NumericType {
    using scalar = T;
    static constexpr int dim = D;
};
template <class F> void dispatch_numeric(ScalarType type, int dim, const F &f) {
    STD_TORCH_CHECK(type == ScalarType::Float || type == ScalarType::Double,
                    "Expected float32 or float64");
    STD_TORCH_CHECK(dim >= 1 && dim <= 3, "Expected dimension 1, 2 or 3");
    if (type == ScalarType::Float) {
        if (dim == 1)
            f(NumericType<float, 1>{});
        else if (dim == 2)
            f(NumericType<float, 2>{});
        else
            f(NumericType<float, 3>{});
    } else {
        if (dim == 1)
            f(NumericType<double, 1>{});
        else if (dim == 2)
            f(NumericType<double, 2>{});
        else
            f(NumericType<double, 3>{});
    }
}
} // namespace sph_native
