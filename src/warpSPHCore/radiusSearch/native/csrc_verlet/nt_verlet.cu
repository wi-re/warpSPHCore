#include "execution/gpu.cuh"
#include "nt_verlet_rows.h"

namespace sph_native::verlet {
namespace {
template<class T,int D> __global__ void partials(Population<T,D> population,T* scratch) {
    Metric<T> local;
    for (int64_t i=int64_t(blockIdx.x)*BlockSize+threadIdx.x;i<population.size;
         i+=int64_t(gridDim.x)*BlockSize) local.merge(population(i));
    __shared__ T shared[BlockSize*3];
    local.store(shared+3*threadIdx.x);
    __syncthreads();
    for (int stride=BlockSize/2;stride;stride/=2) {
        if (threadIdx.x<stride) {
            auto value=Metric<T>::load(shared+3*threadIdx.x);
            value.merge(Metric<T>::load(shared+3*(threadIdx.x+stride)));
            value.store(shared+3*threadIdx.x);
        }
        __syncthreads();
    }
    if (!threadIdx.x) Metric<T>::load(shared).store(scratch+3*blockIdx.x);
}
template<class T> __global__ void decision(const T* scratch,int64_t qb,int64_t rb,
        const T* scale,int support_case,T slack,T* output,bool* flag) {
    Metric<T> query,reference;
    for (int64_t i=threadIdx.x;i<qb;i+=BlockSize) query.merge(Metric<T>::load(scratch+3*i));
    for (int64_t i=threadIdx.x;i<rb;i+=BlockSize) reference.merge(Metric<T>::load(scratch+3*(qb+i)));
    __shared__ T shared[BlockSize*6];
    query.store(shared+6*threadIdx.x); reference.store(shared+6*threadIdx.x+3);
    __syncthreads();
    for (int stride=BlockSize/2;stride;stride/=2) {
        if (threadIdx.x<stride) {
            auto q=Metric<T>::load(shared+6*threadIdx.x),r=Metric<T>::load(shared+6*threadIdx.x+3);
            q.merge(Metric<T>::load(shared+6*(threadIdx.x+stride)));
            r.merge(Metric<T>::load(shared+6*(threadIdx.x+stride)+3));
            q.store(shared+6*threadIdx.x); r.store(shared+6*threadIdx.x+3);
        }
        __syncthreads();
    }
    if (!threadIdx.x) finish(Metric<T>::load(shared),Metric<T>::load(shared+(rb ? 3 : 0)),
                             *scale,support_case,slack,output,flag);
}
struct GpuValidity {
    const Tensors& tensors;
    const ValidityInputs& inputs;
    Tensor partial,output,flag;
    int support_case;
    double slack;
    bool same;
    cudaStream_t stream;
    template<class T,int D,class Launcher>
    void operator()(int64_t queries,RowArgs<T,D> args,const Launcher&) const {
        const auto references=tensors[RX].size(0);
        const auto qb=blocks(queries),rb=same ? 0 : blocks(references);
        auto* scratch=partial.mutable_data_ptr<T>();
        partials<<<int(qb),BlockSize,0,stream>>>(population(args,inputs,queries,false),scratch);
        if (rb) partials<<<int(rb),BlockSize,0,stream>>>(population(args,inputs,references,true),scratch+3*qb);
        decision<<<1,BlockSize,0,stream>>>(scratch,qb,rb,inputs.scale.const_data_ptr<T>(),support_case,T(slack),
                                          output.mutable_data_ptr<T>(),flag.mutable_data_ptr<bool>());
        const auto error=cudaGetLastError();
        if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
};
}
void validity_gpu(const Tensors& tensors,const ValidityInputs& inputs,int support_case,double slack,
                  bool same,const Tensor& partial,const Tensor& output,const Tensor& flag) {
    const stable::accelerator::DeviceGuard guard(tensors[QX].get_device_index());
    const auto stream=current_stream(tensors[QX]);
    execute(GpuValidity{tensors,inputs,partial,output,flag,support_case,slack,same,stream},
            tensors,11,0,6,GpuLauncher{stream});
}
void filter_gpu(const Tensors& tensors,int support,Candidates source,bool collect) {
    const stable::accelerator::DeviceGuard guard(tensors[QX].get_device_index());
    execute(FilterRows{source,collect},tensors,support,0,6,GpuLauncher{current_stream(tensors[QX])});
}
} // namespace sph_native::verlet
