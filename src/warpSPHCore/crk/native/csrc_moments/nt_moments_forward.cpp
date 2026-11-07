#include "crk/native/csrc_preparation/nt_preparation_rows.h"
#include "execution/cpu.h"
#include "execution/dispatch.h"

namespace sph_native {
void crk_moments_forward_gpu(const Tensors&, const crk::PreparationTensors&, int, int, int);
Tensors crk_moments_forward(const OptionalTensors& inputs, const Tensor& volumes,
                           int64_t support, int64_t direction, int64_t kernel) {
    auto t = crk::prepare(inputs, support, direction, kernel);
    crk::volume(volumes, t);
    const auto n = t[QX].size(0), d = t[QX].size(1);
    Tensors moments;
    for (const auto& shape : std::vector<std::vector<int64_t>>{
            {n}, {n,d}, {n,d,d}, {n,d}, {n,d,d}, {n,d,d,d}})
        moments.push_back(stable::new_empty(t[QX], shape));
    const auto counts = stable::new_empty(t[COUNTS], {n});
    const crk::PreparationTensors buffers{volumes, {}, {}, moments, counts};
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        crk_moments_forward_gpu(t, buffers, support, direction, kernel);
#else
        STD_TORCH_CHECK(false, "Native extension was built for CPU only");
#endif
    } else
        execute(crk::PreparationRows<true, crk::SerialPreparationExecutor>{buffers}, t,
                support, direction, 6, CpuLauncher{}, kernel);
    moments.push_back(counts);
    return moments;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native, m) {
    m.def("crk_moments_forward(Tensor?[] inputs, Tensor volumes, int support, int direction, int kernel) -> Tensor[]");
    m.impl("crk_moments_forward", TORCH_BOX(&crk_moments_forward));
}
} // namespace sph_native

#include "nt_moments_jvp.h"
namespace sph_native {
void crk_moments_product_gpu(const Tensors&,const crk_moments_jvp::Buffers&,int,int,int);
Tensors crk_moments_product(const OptionalTensors& inputs,const Tensor& volumes,
        const OptionalTensors& directions,const std::optional<Tensor>& cotangent,
        const Tensors& transpose,int64_t support,int64_t direction,int64_t kernel) {
    auto t=crk::prepare(inputs,support,direction,kernel);crk::volume(volumes,t);
    const auto& x=t[QX];const auto n=x.size(0),m=t[RX].size(0),d=x.size(1);
    const auto width=1+2*d+2*d*d+d*d*d;
    STD_TORCH_CHECK(directions.size()==5,"CRK moment product direction layout");
    for(int k=0;k<5;++k)if(directions[k]) {
        tensor(*directions[k],x,x.scalar_type(),"CRK moment product direction");
        STD_TORCH_CHECK(directions[k]->sizes()==(k==4?volumes.sizes():t[k].sizes()),"CRK moment product direction shape");
    }
    Tensors outputs;
    if(cotangent) {
        tensor(*cotangent,x,x.scalar_type(),"CRK moment product cotangent");
        STD_TORCH_CHECK(cotangent->dim()==2&&cotangent->size(0)==n&&cotangent->size(1)==width,"CRK moment product cotangent shape");
        STD_TORCH_CHECK(transpose.size()==3,"CRK moment product reverse requires CSC");
        vector(transpose[0],x,ScalarType::Long,m,"CSC offsets");vector(transpose[1],x,ScalarType::Int,m,"CSC counts");
        vector(transpose[2],x,ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
        for(const auto& shape:std::vector<std::vector<int64_t>>{{n,d},{m,d},{n},{m},{m},{n,d},{m,d},{n},{m},{m}})
            outputs.push_back(stable::new_zeros(x,shape));
    } else {outputs={stable::new_empty(x,{n,width})};t[OUTPUT]=outputs[0];}
    const crk_moments_jvp::Buffers buffers{directions,transpose,outputs,cotangent?*cotangent:Tensor{},volumes};
    if(x.is_cuda()) {
#ifdef WITH_CUDA
        crk_moments_product_gpu(t,buffers,support,direction,kernel);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    } else execute(crk_moments_jvp::Rows{buffers},t,support,direction,9,CpuLauncher{},kernel);
    return outputs;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("crk_moments_product(Tensor?[] inputs, Tensor volumes, Tensor?[] directions, Tensor? cotangent, Tensor[] transpose, int support, int direction, int kernel) -> Tensor[]");
    m.impl("crk_moments_product",TORCH_BOX(&crk_moments_product));
}
} // namespace sph_native
