#pragma once
#include "execution/dispatch.h"
#include "math/small.h"
#include "execution/numeric.h"
namespace sph_native::mfm {
template<class T,int D> struct GraphRow {
    IndexView offsets,neighbors;
    const int32_t* counts;
    const T *weights,*values;
    T *output,*diagonal;
    SPH_HD void operator()(int64_t i) const {
        auto value=load_vector<T,D>(values,i);SmallVector<T,D> sum{};T diag=0;
        const auto begin=offsets[i];
        for(int32_t local=0;local<counts[i];++local){const auto edge=begin+local;const T weight=weights[edge];
            if(weight>T(0)){sum+=weight*(value-load_vector<T,D>(values,neighbors[edge]));diag+=weight;}}
        store_vector(output,i,sum);diagonal[i]=diag;
    }
};
template<class T> struct SegmentRow {
    const int64_t *permutation,*starts,*ends;
    const T* values;T* output;int64_t width;
    SPH_HD void operator()(int64_t row) const {
        const auto target=row/width,component=row%width;T sum=output[row];
        for(auto index=starts[target];index<ends[target];++index)sum+=values[permutation[index]*width+component];
        output[row]=sum;
    }
};
struct GraphBuffers {Tensor offsets,counts,neighbors,weights,values,output,diagonal;};
struct SegmentBuffers {Tensor permutation,starts,ends,values,output;};
template<class Launch> void graph(const GraphBuffers& b,const Launch& launch) {
    dispatch_numeric(b.values.scalar_type(),int(b.values.size(1)),[&](auto tag){
        using T=typename decltype(tag)::scalar;constexpr int D=decltype(tag)::dim;
        launch(b.values.size(0),GraphRow<T,D>{index_view(b.offsets),index_view(b.neighbors),b.counts.const_data_ptr<int32_t>(),
            b.weights.const_data_ptr<T>(),b.values.const_data_ptr<T>(),b.output.mutable_data_ptr<T>(),b.diagonal.mutable_data_ptr<T>()});});
}
template<class Launch> void segment(const SegmentBuffers& b,const Launch& launch) {
    dispatch_numeric(b.values.scalar_type(),1,[&](auto tag){using T=typename decltype(tag)::scalar;
        launch(b.output.numel(),SegmentRow<T>{b.permutation.const_data_ptr<int64_t>(),b.starts.const_data_ptr<int64_t>(),b.ends.const_data_ptr<int64_t>(),
            b.values.const_data_ptr<T>(),b.output.mutable_data_ptr<T>(),b.values.size(1)});});
}
void graph_gpu(const GraphBuffers&);void segment_gpu(const SegmentBuffers&);
}
