#include "nt_polyfit_rows.h"
#include "torch/validation.h"
#include "execution/cpu.h"

namespace sph_native::polyfit {
Tensors prepare(const OptionalTensors& inputs,const Tensor& exponents,const Tensor& entries,
                const Tensor& volumes,const Tensor& values,int direction,int kernel,int sector,int sectors,bool subtract) {
    STD_TORCH_CHECK(inputs.size()==OUTPUT,"Invalid polynomial input layout");
    Tensors flat;for (const auto& value:inputs) flat.push_back(value ? *value:Tensor{});
    auto tensors=prepare_operator(flat,11,direction);
    const auto& x=tensors[QX]; const auto n=x.size(0),nr=tensors[RX].size(0),d=x.size(1);
    vector(tensors[DENSITY],x,x.scalar_type(),nr,"source densities");
    tensor(exponents,x,ScalarType::Int,"polynomial exponents");
    STD_TORCH_CHECK(exponents.dim()==2 && exponents.size(0)>0 && exponents.size(1)==d,"Exponents must have shape (basis,dimension)");
    tensor(entries,x,ScalarType::Int,"packed moment entries");
    STD_TORCH_CHECK(entries.dim()==2 && entries.size(0)==exponents.size(0)*(exponents.size(0)+1)/2 && entries.size(1)==2,"Invalid packed moment layout");
    STD_TORCH_CHECK(kernel>=0 && kernel<=42 && sectors>0 && sector<sectors && (sector<0 || d<3),"Invalid polynomial stencil");
    tensor(volumes,x,x.scalar_type(),"source volumes");
    STD_TORCH_CHECK(volumes.dim()==1 && (volumes.numel()==0 || volumes.numel()==nr),"Source volume shape mismatch");
    tensor(values,x,x.scalar_type(),"polynomial field");
    if (values.dim()==2) {
        STD_TORCH_CHECK(values.size(0)==nr && (!subtract || nr==n),"Polynomial field population mismatch");
    } else { STD_TORCH_CHECK(values.numel()==0 && !subtract,"Expected (sources,components) values"); }
    return tensors;
}
Buffers buffers(const Tensors& t,const Tensor& exponents,const Tensor& entries,const Tensor& volumes,
                const Tensor& values,int sector,int sectors,bool unit,bool subtract) {
    Buffers b{};
    b.exponents=exponents;b.entries=entries;b.volumes=volumes.numel() ? volumes:Tensor{};
    b.rhs=values.dim()==2;b.values=b.rhs ? values:Tensor{};
    b.basis=int(exponents.size(0));b.components=b.rhs ? int(values.size(1)):1;
    b.sector=sector;b.sectors=sectors;b.unit=unit;b.subtract=subtract;
    return b;
}
void run(const Tensors& t,const Buffers& b,int direction,int kernel,int role) {
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        run_gpu(t,b,direction,kernel,role);
#else
        STD_TORCH_CHECK(false,"Build native GPU support to use CUDA/HIP");
#endif
    } else execute(Dispatch<CpuLauncher>{b,role},t,11,direction,1,CpuLauncher{},kernel);
}
std::tuple<Tensor,std::optional<Tensor>> forward(const OptionalTensors& inputs,const Tensor& exponents,const Tensor& entries,
        const Tensor& volumes,const Tensor& values,int64_t direction,int64_t kernel,int64_t sector,int64_t sectors,bool unit,bool subtract) {
    auto t=prepare(inputs,exponents,entries,volumes,values,int(direction),int(kernel),int(sector),int(sectors),subtract);
    auto b=buffers(t,exponents,entries,volumes,values,int(sector),int(sectors),unit,subtract);
    const int64_t width=b.rhs ? int64_t(b.basis)*b.components:entries.size(0);
    b.output=stable::new_empty(t[QX],{t[QX].size(0),width});
    if (!b.rhs) b.counts=stable::new_zeros(t[QX],{t[QX].size(0)},ScalarType::Int);
    run(t,b,int(direction),int(kernel),0);
    return {b.output,b.counts.defined() ? std::optional<Tensor>(b.counts):std::nullopt};
}
Tensor jvp(const OptionalTensors& inputs,const Tensor& exponents,const Tensor& entries,
        const Tensor& volumes,const Tensor& values,const OptionalTensors& tangents,
        int64_t direction,int64_t kernel,int64_t sector,int64_t sectors,bool unit,bool subtract) {
    auto t=prepare(inputs,exponents,entries,volumes,values,int(direction),int(kernel),int(sector),int(sectors),subtract);
    auto b=buffers(t,exponents,entries,volumes,values,int(sector),int(sectors),unit,subtract);
    STD_TORCH_CHECK(tangents.size()==8,"Invalid polynomial tangent layout");
    const Tensors primals={t[QX],t[RX],t[QH],t[MASS],t[DENSITY],volumes,values,t[LO]};
    for (int slot=0;slot<8;++slot) {
        Tensor tangent=tangents[slot] ? *tangents[slot]:Tensor{};
        if (tangent.defined()) {
            tensor(tangent,t[QX],t[QX].scalar_type(),"polynomial tangent");
            STD_TORCH_CHECK(tangent.sizes()==primals[slot].sizes(),"Polynomial tangent shape mismatch");
        }
        b.tangents.push_back(tangent);
    }
    const int64_t width=b.rhs ? int64_t(b.basis)*b.components:entries.size(0);
    b.output=stable::new_empty(t[QX],{t[QX].size(0),width});
    run(t,b,int(direction),int(kernel),1);return b.output;
}
Tensors vjp(const OptionalTensors& inputs,const Tensor& exponents,const Tensor& entries,
        const Tensor& volumes,const Tensor& values,const Tensor& cotangent,const Tensors& transpose,
        int64_t direction,int64_t kernel,int64_t sector,int64_t sectors,bool unit,bool subtract) {
    auto t=prepare(inputs,exponents,entries,volumes,values,int(direction),int(kernel),int(sector),int(sectors),subtract);
    auto b=buffers(t,exponents,entries,volumes,values,int(sector),int(sectors),unit,subtract);
    tensor(cotangent,t[QX],t[QX].scalar_type(),"polynomial cotangent");
    const auto n=t[QX].size(0),nr=t[RX].size(0),d=t[QX].size(1);
    const int64_t width=b.rhs ? int64_t(b.basis)*b.components:entries.size(0);
    STD_TORCH_CHECK(cotangent.dim()==2 && cotangent.size(0)==n && cotangent.size(1)==width,"Polynomial cotangent shape mismatch");
    STD_TORCH_CHECK(transpose.size()==3,"Polynomial pullback requires CSC");
    vector(transpose[0],t[QX],ScalarType::Long,nr,"CSC offsets");
    vector(transpose[1],t[QX],ScalarType::Int,nr,"CSC counts");
    vector(transpose[2],t[QX],ScalarType::Long,t[NEIGHBORS].numel(),"CSC queries");
    b.transposeOffsets=index_view(transpose[0]);b.transposeCounts=transpose[1].const_data_ptr<int32_t>();b.transposeQueries=index_view(transpose[2]);
    b.cotangent=cotangent;
    for (const auto& primal:Tensors{t[QX],t[RX],t[QH],t[MASS],t[DENSITY],volumes,values})
        b.bars.push_back(stable::new_zeros(primal,primal.sizes()));
    b.bars.push_back(stable::new_zeros(t[QX],{n,d}));
    run(t,b,int(direction),int(kernel),2);
    run(t,b,int(direction),int(kernel),3);
    return b.bars;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("polyfit_forward(Tensor?[] inputs, Tensor exponents, Tensor entries, Tensor volumes, Tensor values, int direction, int kernel, int sector, int sectors, bool unit, bool subtract) -> (Tensor,Tensor?)");
    m.impl("polyfit_forward",TORCH_BOX(&forward));
    m.def("polyfit_jvp(Tensor?[] inputs, Tensor exponents, Tensor entries, Tensor volumes, Tensor values, Tensor?[] tangents, int direction, int kernel, int sector, int sectors, bool unit, bool subtract) -> Tensor");
    m.impl("polyfit_jvp",TORCH_BOX(&jvp));
    m.def("polyfit_vjp(Tensor?[] inputs, Tensor exponents, Tensor entries, Tensor volumes, Tensor values, Tensor cotangent, Tensor[] transpose, int direction, int kernel, int sector, int sectors, bool unit, bool subtract) -> Tensor[]");
    m.impl("polyfit_vjp",TORCH_BOX(&vjp));
}
} // namespace sph_native::polyfit
