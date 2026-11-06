#pragma once
#include "execution/fieldTraversal.h"
#include "kernels/evaluation.h"
#include "nt_curl_kernel.h"
namespace sph_native {
template <class T, int D> struct CurlPair {
    template <class Args>
    SPH_HD static void accumulate(const Args &a, int64_t i, int64_t j, const T (&dx)[D], T r, T w,
                                  T g, T (&sum)[D * D * D]) {
        T gradient[D]={};
        if constexpr(D>1)curl_kernel_gradient(a,i,j,dx,r,gradient);
        if constexpr (D == 2) {
            for (int c = 0; c < a.outputs; ++c)
                sum[c] += gradient[0] * field_weight(a, i, j, a.outputs + c) -
                          gradient[1] * field_weight(a, i, j, c);
        }
        if constexpr (D == 3) {
            int stride = a.fields / 3;
            for (int c = 0; c < stride; ++c) {
                sum[c] += gradient[1] * field_weight(a, i, j, 2 * stride + c) -
                          gradient[2] * field_weight(a, i, j, stride + c);
                sum[stride + c] += gradient[2] * field_weight(a, i, j, c) -
                                   gradient[0] * field_weight(a, i, j, 2 * stride + c);
                sum[2 * stride + c] += gradient[0] * field_weight(a, i, j, stride + c) -
                                       gradient[1] * field_weight(a, i, j, c);
            }
        }
    }
};
template <class T, int D> using CurlRow = FieldOperatorRow<T, D, CurlPair<T, D>>;
struct CurlRows {
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        launch(n, CurlRow<T, D>{a});
    }
};
} // namespace sph_native
