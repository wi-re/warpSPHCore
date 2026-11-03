#pragma once
#include "execution/numeric.h"
#include "math/distance.h"
namespace sph_native::block {
template<class T> struct ApplyRow {
    const T *inverse,*values;
    T* output;
    int dimension;
    int64_t inputStride,outputStride;
    bool transpose;
    SPH_HD void operator()(int64_t row) const {
        const int64_t particle=row/dimension;
        const int component=int(row%dimension);
        T value=0;
        for (int k=0;k<dimension;++k) {
            const int at=transpose ? k*dimension+component:component*dimension+k;
            value+=inverse[particle*dimension*dimension+at]*values[(particle*dimension+k)*inputStride];
        }
        output[row*outputStride]=value;
    }
};
template<class Launch> void execute(const Tensor& inverse,const Tensor& values,const Tensor& output,bool transpose,const Launch& launch) {
    dispatch_numeric(values.scalar_type(),int(inverse.size(1)),[&](auto tag) {
        using T=typename decltype(tag)::scalar;
        launch(values.numel(),ApplyRow<T>{inverse.const_data_ptr<T>(),values.const_data_ptr<T>(),
                                         output.mutable_data_ptr<T>(),int(inverse.size(1)),values.stride(0),output.stride(0),transpose});
    });
}
void apply_gpu(const Tensor&,const Tensor&,const Tensor&,bool);
} // namespace sph_native::block
