#include "nt_verlet_rows.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include "execution/scan.h"
#include "api.h"

namespace sph_native::verlet {
namespace {
Tensors geometry_inputs(const Tensors& inputs, int support) {
    STD_TORCH_CHECK(inputs.size()==7,"Expected seven geometry tensors");
    Tensors result(SLOTS);
    for (int k=0;k<4;++k) result[k]=inputs[k];
    result[LO]=inputs[4]; result[HI]=inputs[5]; result[PERIODIC]=inputs[6];
    geometry(result,support);
    return result;
}
struct CpuValidity {
    const Tensors& tensors;
    const ValidityInputs& inputs;
    Tensor partial, output, flag;
    int support_case;
    double slack;
    bool same;
    template <class T,int D,class Launcher>
    void operator()(int64_t queries, RowArgs<T,D> args, const Launcher& launch) const {
        const auto references=tensors[RX].size(0);
        const auto qb=blocks(queries), rb=same ? 0 : blocks(references);
        T* scratch=partial.mutable_data_ptr<T>();
        const auto reduce=[&](bool reference, int64_t count, int64_t offset, int64_t groups) {
            const auto pop=population(args,inputs,count,reference);
            launch(groups,[=](int64_t group) {
                Metric<T> summary;
                for (int64_t begin=group*BlockSize;begin<count;begin+=groups*BlockSize)
                    for (int64_t i=begin;i<std::min(begin+BlockSize,count);++i) summary.merge(pop(i));
                summary.store(scratch+3*(offset+group));
            });
        };
        reduce(false,queries,0,qb);
        if (!same) reduce(true,references,qb,rb);
        Metric<T> query,reference;
        for (int64_t i=0;i<qb;++i) query.merge(Metric<T>::load(scratch+3*i));
        for (int64_t i=0;i<rb;++i) reference.merge(Metric<T>::load(scratch+3*(qb+i)));
        finish(query,same ? query : reference,*inputs.scale.const_data_ptr<T>(),support_case,T(slack),
               output.mutable_data_ptr<T>(),flag.mutable_data_ptr<bool>());
    }
};
}
std::tuple<Tensor,Tensor> metrics(const Tensors& geometry, const Tensor& prior_q,
        const Tensor& prior_r, const Tensor& old_qh,const Tensor& old_rh,const Tensor& scale,
        int64_t support_case,double slack,bool same) {
    auto tensors=geometry_inputs(geometry,11);
    const auto& q=tensors[QX]; const auto& r=tensors[RX];
    tensor(prior_q,q,q.scalar_type(),"prior query positions");
    tensor(prior_r,q,q.scalar_type(),"prior reference positions");
    STD_TORCH_CHECK(prior_q.dim()==2 && prior_r.dim()==2 && prior_q.size(0)==q.size(0) && prior_r.size(0)==r.size(0) && prior_q.size(1)==q.size(1) && prior_r.size(1)==r.size(1),"Prior population shape mismatch");
    vector(old_qh,q,q.scalar_type(),q.size(0),"prior query supports");
    vector(old_rh,q,q.scalar_type(),r.size(0),"prior reference supports");
    vector(scale,q,q.scalar_type(),1,"Verlet scale");
    STD_TORCH_CHECK(support_case>=0 && support_case<=2,"Invalid support case");
    STD_TORCH_CHECK(q.size(0)>0 && r.size(0)>0,"Empty validity checks are handled by the wrapper");
    auto partial=stable::new_empty(q,{blocks(q.size(0))+(same ? 0 : blocks(r.size(0))),3});
    auto output=stable::new_empty(q,{4});
    auto flag=stable::new_empty(q,{},ScalarType::Bool);
    const ValidityInputs inputs{prior_q,prior_r,old_qh,old_rh,scale};
    if (q.is_cuda()) {
#ifdef WITH_CUDA
        validity_gpu(tensors,inputs,int(support_case),slack,same,partial,output,flag);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else execute(CpuValidity{tensors,inputs,partial,output,flag,int(support_case),slack,same},
                   tensors,11,0,6,CpuLauncher{});
    return {flag,output};
}

std::tuple<Tensor,Tensor,Tensor> filter(const Tensors& inputs,const Tensor& neighbors,
                                      const Tensor& counts,const Tensor& offsets,int64_t support) {
    auto tensors=geometry_inputs(inputs,int(support));
    const auto& q=tensors[QX];
    vector(counts,q,ScalarType::Int,q.size(0),"candidate counts");
    for (const auto& index : {neighbors,offsets}) {
        STD_TORCH_CHECK(index.scalar_type()==ScalarType::Int || index.scalar_type()==ScalarType::Long,"CSR storage must be int32/int64");
        tensor(index,q,index.scalar_type(),"candidate indices");
        STD_TORCH_CHECK(index.dim()==1,"CSR storage must be one dimensional");
    }
    STD_TORCH_CHECK(offsets.size(0)==q.size(0),"CSR offsets must have one entry per query");
    const Candidates source{index_view(neighbors),index_view(offsets),counts.const_data_ptr<int32_t>()};
    tensors[COUNTS]=stable::new_empty(q,{q.size(0)},ScalarType::Int);
    tensors[OFFSETS]=stable::new_empty(q,{q.size(0)},ScalarType::Long);
    auto launch=[&](bool collect) {
        if (q.is_cuda()) {
#ifdef WITH_CUDA
            filter_gpu(tensors,int(support),source,collect);
#else
            STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
        } else execute(FilterRows{source,collect},tensors,int(support),0,6,CpuLauncher{});
    };
    launch(false);
    int64_t total=0;
    if (q.is_cuda()) {
#ifdef WITH_CUDA
        Context context{};
        context.device_index=q.get_device_index();
#if defined(USE_ROCM)
        context.device=Device::HIP;
#else
        context.device=Device::CUDA;
#endif
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(context.device_index,&context.stream));
        total=runtime::gpu_scan(context,q.size(0),{tensors[COUNTS].const_data_ptr(),q.size(0)},
                                {tensors[OFFSETS].mutable_data_ptr(),q.size(0)});
#endif
    } else total=exclusive_scan_cpu(tensors[COUNTS].const_data_ptr<int32_t>(),
                                    tensors[OFFSETS].mutable_data_ptr<int64_t>(),q.size(0));
    tensors[NEIGHBORS]=stable::new_empty(q,{total},ScalarType::Long);
    launch(true);
    return {tensors[NEIGHBORS],tensors[COUNTS],tensors[OFFSETS]};
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("verlet_metrics(Tensor[] geometry, Tensor prior_q, Tensor prior_r, Tensor old_qh, Tensor old_rh, Tensor scale, int support_case, float slack, bool same) -> (Tensor, Tensor)");
    m.impl("verlet_metrics",TORCH_BOX(&metrics));
    m.def("verlet_filter(Tensor[] geometry, Tensor neighbors, Tensor counts, Tensor offsets, int support) -> (Tensor, Tensor, Tensor)");
    m.impl("verlet_filter",TORCH_BOX(&filter));
}
} // namespace sph_native::verlet
