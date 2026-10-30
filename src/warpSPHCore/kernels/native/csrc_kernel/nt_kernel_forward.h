#pragma once
#include "kernels/evaluation.h"
#include "execution/numeric.h"
namespace sph_native {
template <class T, int D> struct KernelRow {
    const T *x, *h;
    int kernel;
    T *value, *gradient, *hessian, *dh, *gradient_dh, *laplacian, *radial;
    SPH_HD void operator()(int64_t i) const {
        auto e = evaluate_kernel<T, D>(x + i * D, h[i], kernel);
        value[i] = e.value;
        dh[i] = e.dh;
        laplacian[i] = e.laplacian;
        for (int d = 0; d < D; ++d) {
            gradient[i * D + d] = e.gradient[d];
            gradient_dh[i * D + d] = e.gradient_dh[d];
        }
        for (int d = 0; d < D * D; ++d)
            hessian[i * D * D + d] = e.hessian[d];
        T *out = radial + i * 7;
        out[0] = e.shape.v;
        out[1] = e.shape.d1;
        out[2] = e.shape.d2;
        out[3] = e.shape.d3;
        out[4] = normalization<T, D>(kernel);
        out[5] = kernel_scale<T, D>(kernel);
        out[6] = packing_ratio<T>(kernel);
    }
};
template <class Launcher>
void kernel_execute(const Tensors &t, int kernel, const Launcher &launch) {
    dispatch_numeric(t[0].scalar_type(), int(t[0].size(1)), [&](auto tag) {
        using T = typename decltype(tag)::scalar;
        constexpr int D = decltype(tag)::dim;
        KernelRow<T, D> row{
            t[0].const_data_ptr<T>(),   t[1].const_data_ptr<T>(),   kernel,
            t[2].mutable_data_ptr<T>(), t[3].mutable_data_ptr<T>(), t[4].mutable_data_ptr<T>(),
            t[5].mutable_data_ptr<T>(), t[6].mutable_data_ptr<T>(), t[7].mutable_data_ptr<T>(),
            t[8].mutable_data_ptr<T>()};
        launch(t[0].size(0), row);
    });
}
} // namespace sph_native
