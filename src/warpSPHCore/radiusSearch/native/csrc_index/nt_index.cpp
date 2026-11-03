#include "nt_index.h"
#include "torch/validation.h"
#include "execution/cpu.h"
#include <cstring>
namespace sph_native {
void launch_radius(RadiusTask,const Tensors&,int,compact::View);
}
namespace sph_native::search_index {
Tensors geometry_inputs(const Tensors& inputs,int support) {
    STD_TORCH_CHECK(inputs.size()==7,"Expected query/reference/support/domain geometry");
    Tensors t(SLOTS);for(int k=0;k<4;++k)t[k]=inputs[k];
    t[LO]=inputs[4];t[HI]=inputs[5];t[PERIODIC]=inputs[6];geometry(t,support);return t;
}
Tensors build(const Tensors& inputs,const Tensor& metadata,int64_t support) {
    auto t=geometry_inputs(inputs,int(support));const auto& x=t[QX];
    STD_TORCH_CHECK(metadata.device().is_cpu() && metadata.scalar_type()==ScalarType::Double && metadata.is_contiguous() && metadata.numel()==MetadataSize,"Expected fixed CPU index metadata");
    auto buffer=[](const Tensor& value){return ConstBuffer{value.const_data_ptr(),value.numel()};};
    Problem problem{};problem.precision=x.scalar_type()==ScalarType::Float ? Precision::Float32:Precision::Float64;
    problem.dimension=int(x.size(1));problem.query={x.size(0),buffer(x),buffer(t[QH])};
    problem.reference={t[RX].size(0),buffer(t[RX]),buffer(t[RH])};problem.domain={buffer(t[LO]),buffer(t[HI]),buffer(t[PERIODIC])};problem.properties.support=Support(support);
    Context context{};
    if(x.is_cuda()) {
#ifdef WITH_CUDA
#if defined(USE_ROCM)
        context.device=Device::HIP;
#else
        context.device=Device::CUDA;
#endif
        context.device_index=x.get_device_index();STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(context.device_index,&context.stream));
#endif
    }
    auto index=build_compact_hash(context,problem);const auto& data=compact::Access::get(index);
    static_assert(sizeof(compact::ParticleKey)==3*sizeof(int64_t));
    static_assert(sizeof(compact::Cell)==4*sizeof(int64_t));static_assert(sizeof(compact::Bucket)==2*sizeof(int64_t));
    Tensors arrays={stable::new_empty(x,{data.references,3},ScalarType::Long),
                   stable::new_empty(x,{data.occupied,4},ScalarType::Long),
                   stable::new_empty(x,{data.grid.buckets,2},ScalarType::Long)};
    const void* sources[]={data.particles,data.cells,data.buckets};
    for(int k=0;k<3;++k)if(arrays[k].numel()) {
        const size_t bytes=size_t(arrays[k].numel())*sizeof(int64_t);
        if(x.is_cuda()) {
#ifdef WITH_CUDA
            copy_gpu(arrays[k],sources[k],bytes);
#endif
        } else std::memcpy(arrays[k].mutable_data_ptr(),sources[k],bytes);
    }
    auto* out=metadata.mutable_data_ptr<double>();out[Version]=1;out[Dimension]=data.dimension;
    out[PrecisionBits]=x.scalar_type()==ScalarType::Float ? 32:64;out[SupportMode]=support;out[References]=data.references;
    out[Occupied]=data.occupied;out[Buckets]=data.grid.buckets;out[CellWidth]=data.grid.width;out[RadiusBound]=data.radius_bound;
    for(int c=0;c<3;++c){out[Lower+c]=data.grid.lower[c];out[Upper+c]=data.grid.upper[c];out[Cells+c]=double(data.grid.cells[c]);out[Periodic+c]=data.grid.periodic[c] ? 1:0;}
    // The construction index is released here. Its arrays were copied only
    // device-to-device; particle data and neighbor lists never visit the host.
    return arrays;
}
compact::View view(const Tensors& arrays,const Tensor& metadata,const Tensor& x,int support) {
    STD_TORCH_CHECK(arrays.size()==3 && metadata.device().is_cpu() && metadata.scalar_type()==ScalarType::Double && metadata.is_contiguous() && metadata.numel()==MetadataSize,"Invalid retained index layout");
    const auto* m=metadata.const_data_ptr<double>();
    STD_TORCH_CHECK(m[Version]==1 && m[Dimension]==x.size(1) && m[PrecisionBits]==(x.scalar_type()==ScalarType::Float ? 32:64) && m[SupportMode]==support,"Index geometry policy mismatch");
    for(int k=0;k<3;++k){tensor(arrays[k],x,ScalarType::Long,"index array");STD_TORCH_CHECK(arrays[k].dim()==2 && arrays[k].size(0)==int64_t(m[k==0 ? References:k==1 ? Occupied:Buckets]) && arrays[k].size(1)==(k==0 ? 3:k==1 ? 4:2),"Index array shape mismatch");}
    compact::View result{};result.grid.width=m[CellWidth];result.grid.buckets=int64_t(m[Buckets]);
    for(int c=0;c<3;++c){result.grid.lower[c]=m[Lower+c];result.grid.upper[c]=m[Upper+c];result.grid.cells[c]=int64_t(m[Cells+c]);result.grid.periodic[c]=m[Periodic+c]!=0;}
    result.particles=reinterpret_cast<const compact::ParticleKey*>(arrays[0].const_data_ptr<int64_t>());
    result.cells=reinterpret_cast<const compact::Cell*>(arrays[1].const_data_ptr<int64_t>());
    result.buckets=reinterpret_cast<const compact::Bucket*>(arrays[2].const_data_ptr<int64_t>());result.occupied=int64_t(m[Occupied]);return result;
}
void search(const Tensors& inputs,const Tensor& metadata,const Tensors& arrays,const Tensor& counts,
            const std::optional<Tensor>& offsets,const std::optional<Tensor>& neighbors,int64_t support) {
    auto t=geometry_inputs(inputs,int(support));const auto& x=t[QX];
    auto index=view(arrays,metadata,x,int(support));
    STD_TORCH_CHECK(t[RX].size(0)==arrays[0].size(0),"Index source population mismatch");
    vector(counts,x,ScalarType::Int,x.size(0),"target counts");t[COUNTS]=counts;
    const bool collect=offsets.has_value();STD_TORCH_CHECK(collect==neighbors.has_value(),"Collection needs offsets and neighbors");
    if(collect) {
        STD_TORCH_CHECK((offsets->scalar_type()==ScalarType::Int || offsets->scalar_type()==ScalarType::Long) &&
                        (neighbors->scalar_type()==ScalarType::Int || neighbors->scalar_type()==ScalarType::Long),"CSR storage must be int32/int64");
        vector(*offsets,x,offsets->scalar_type(),x.size(0),"target offsets");tensor(*neighbors,x,neighbors->scalar_type(),"target neighbors");
        STD_TORCH_CHECK(neighbors->dim()==1,"Expected a target neighbor vector");t[OFFSETS]=*offsets;t[NEIGHBORS]=*neighbors;
    }
    launch_radius(collect ? RadiusTask::Collect:RadiusTask::Count,t,int(support),index);
}
void rows(const Tensor& offsets,const Tensor& counts,const Tensor& output,int64_t start) {
    STD_TORCH_CHECK(start>=0 && (output.scalar_type()==ScalarType::Int || output.scalar_type()==ScalarType::Long) &&
                    (offsets.scalar_type()==ScalarType::Int || offsets.scalar_type()==ScalarType::Long),"Invalid global-row storage");
    vector(counts,output,ScalarType::Int,counts.numel(),"row counts");vector(offsets,output,offsets.scalar_type(),counts.numel(),"row offsets");
    tensor(output,output,output.scalar_type(),"global rows");STD_TORCH_CHECK(output.dim()==1,"Global rows must be a vector");
    if(output.is_cuda()) {
#ifdef WITH_CUDA
        fill_gpu(offsets,counts,output,start);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else fill_rows(offsets,counts,output,start,CpuLauncher{});
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("search_index(Tensor[] geometry, Tensor(a!) metadata, int support) -> Tensor[]");m.impl("search_index",TORCH_BOX(&build));
    m.def("search_index_apply(Tensor[] geometry, Tensor metadata, Tensor[] arrays, Tensor(a!) counts, Tensor? offsets, Tensor(b!)? neighbors, int support) -> ()");m.impl("search_index_apply",TORCH_BOX(&search));
    m.def("csr_global_rows(Tensor offsets, Tensor counts, Tensor(a!) rows, int start) -> ()");m.impl("csr_global_rows",TORCH_BOX(&rows));
}
}
