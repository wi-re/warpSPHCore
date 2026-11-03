#pragma once
#include "execution/adjointTraversal.h"
#include "torch/validation.h"
namespace sph_native {
template <class Pair> struct AdjointRows {
    const Tensors &gradients;
    const Tensor &cotangent;
    int64_t mask;
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        GradientBuffers<T> buffers{};
        buffers.mask = mask;
        for (int slot = 0; slot < OUTPUT; ++slot)
            buffers.pointers[slot] = gradients[slot].mutable_data_ptr<T>();
        launch(n, AdjointRow<T, D, typename Pair::template type<T, D>>{
                      a, cotangent.const_data_ptr<T>(), buffers});
    }
};
inline Tensors gradient_buffers(const Tensors &t, int64_t mask) {
    Tensors gradients;
    for (int slot = 0; slot < OUTPUT; ++slot) {
        if (mask & (int64_t(1) << slot))
            gradients.push_back(stable::new_zeros(t[slot], t[slot].sizes()));
        else
            gradients.push_back(stable::new_empty(t[QX], {0}));
    }
    return gradients;
}
} // namespace sph_native
