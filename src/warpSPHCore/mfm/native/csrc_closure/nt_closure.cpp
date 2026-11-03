#include "nt_closure_rows.h"
#include "torch/validation.h"
#include "execution/cpu.h"
namespace sph_native::mfm {
std::tuple<Tensor,Tensor> graph_apply(const Tensor& offsets,const Tensor& counts,const Tensor& neighbors,const Tensor& weights,const Tensor& values) {
    tensor(values,values,values.scalar_type(),"graph values");
    STD_TORCH_CHECK(values.dim()==2 && values.size(1)>=1 && values.size(1)<=3,"Graph values must be (n,d), d<=3");
    const auto n=values.size(0);vector(counts,values,ScalarType::Int,n,"graph counts");
    STD_TORCH_CHECK(offsets.scalar_type()==ScalarType::Int || offsets.scalar_type()==ScalarType::Long,"Graph offsets must be int32/int64");
    vector(offsets,values,offsets.scalar_type(),n,"graph offsets");
    STD_TORCH_CHECK(neighbors.scalar_type()==ScalarType::Int || neighbors.scalar_type()==ScalarType::Long,"Graph neighbors must be int32/int64");
    tensor(neighbors,values,neighbors.scalar_type(),"graph neighbors");STD_TORCH_CHECK(neighbors.dim()==1,"Graph neighbors must be a vector");
    vector(weights,values,values.scalar_type(),neighbors.numel(),"graph weights");
    GraphBuffers b{offsets,counts,neighbors,weights,values,stable::empty_like(values),stable::new_empty(values,{n})};
    if(values.is_cuda()) {
#ifdef WITH_CUDA
        graph_gpu(b);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else graph(b,CpuLauncher{});
    return {b.output,b.diagonal};
}
void segment_sum(const Tensor& permutation,const Tensor& starts,const Tensor& ends,const Tensor& values,const Tensor& output) {
    tensor(values,values,values.scalar_type(),"segment values");tensor(output,values,values.scalar_type(),"segment output");
    STD_TORCH_CHECK(values.dim()==2 && output.dim()==2 && values.size(1)==output.size(1),"Segment width mismatch");
    vector(permutation,values,ScalarType::Long,values.size(0),"segment permutation");
    vector(starts,values,ScalarType::Long,output.size(0),"segment starts");vector(ends,values,ScalarType::Long,output.size(0),"segment ends");
    SegmentBuffers b{permutation,starts,ends,values,output};
    if(values.is_cuda()) {
#ifdef WITH_CUDA
        segment_gpu(b);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else segment(b,CpuLauncher{});
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("meshless_graph(Tensor offsets, Tensor counts, Tensor neighbors, Tensor weights, Tensor values) -> (Tensor,Tensor)");m.impl("meshless_graph",TORCH_BOX(&graph_apply));
    m.def("segment_sum(Tensor permutation, Tensor starts, Tensor ends, Tensor values, Tensor(a!) output) -> ()");m.impl("segment_sum",TORCH_BOX(&segment_sum));
}
}
