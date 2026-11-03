#include "nt_differential_rows.h"
#include "execution/cpu.h"
namespace sph_native {
void corrected_differential_gpu(const Tensors&,const OptionalTensors&,const differential::Buffers&,
    int,int,int,int,int,int,bool,int);
namespace {
Tensors prepare_differential(const OptionalTensors& flat,const Tensor& example,
        int operation,int support,int direction,int kernel,int scheme,int lap,bool dot,bool consistent) {
    using namespace differential;
    STD_TORCH_CHECK(flat.size()==Slots,"invalid corrected derivative input layout");
    STD_TORCH_CHECK(valid_kernel(kernel),"Unknown native kernel");
    OptionalTensors inputs(flat.begin(),flat.begin()+OUTPUT),corrections(flat.begin()+OUTPUT,flat.end());
    if (operation==6) {
        Tensors borrowed; for (const auto& value:inputs) borrowed.push_back(value ? *value:Tensor{});
        auto t=prepare_operator(borrowed,support,direction);
        vector(example,t[QX],t[QX].scalar_type(),t[QX].size(0),"density derivative output");
        t[OUTPUT]=example;
        return t;
    }
    if (operation<=7)
        return prepare_corrected_operator(inputs,corrections,support,direction,operation,scheme,lap,dot,consistent,&example);
    Tensors borrowed; for (const auto& value:inputs) borrowed.push_back(value ? *value:Tensor{});
    auto t=prepare_pair_geometry(borrowed,support,direction);
    const auto n=t[QX].size(0),m=t[RX].size(0),d=t[QX].size(1);
    STD_TORCH_CHECK(operation>=Volume && operation<=Consistency,"Unknown preparation derivative");
    if (operation!=Volume) STD_TORCH_CHECK(flat[VOLUME],"reference volumes are required");
    if (operation!=Volume) vector(*flat[VOLUME],t[QX],t[QX].scalar_type(),m,"reference volumes");
    if (operation==Consistency) {
        STD_TORCH_CHECK(flat[A] && flat[B],"Consistency density requires A and B");
        vector(t[MASS],t[QX],t[QX].scalar_type(),m,"reference masses");
        vector(*flat[A],t[QX],t[QX].scalar_type(),n,"CRK A");
        tensor(*flat[B],t[QX],t[QX].scalar_type(),"CRK B");
        STD_TORCH_CHECK(flat[B]->dim()==2 && flat[B]->size(0)==n && flat[B]->size(1)==d,"CRK B shape");
    }
    tensor(example,t[QX],t[QX].scalar_type(),"preparation derivative output");
    const int columns=operation==Moments ? 1+2*d+2*d*d+d*d*d : 1;
    STD_TORCH_CHECK(example.dim()==2 && example.size(0)==n && example.size(1)==columns,"preparation output shape");
    t[OUTPUT]=example;
    return t;
}
void launch_differential(const Tensors& t,const OptionalTensors& flat,const differential::Buffers& buffers,
        int operation,int support,int direction,int kernel,int scheme,int lap,bool dot,int mode) {
    if (t[QX].is_cuda()) {
#ifdef WITH_CUDA
        corrected_differential_gpu(t,flat,buffers,operation,support,direction,kernel,scheme,lap,dot,mode);
#else
        STD_TORCH_CHECK(false,"Native extension was built for CPU only");
#endif
    } else if (mode==0) execute(differential::Rows<0>{flat,buffers},t,support,direction,operation,CpuLauncher{},kernel,scheme,lap,dot);
    else if (mode==1) execute(differential::Rows<1>{flat,buffers},t,support,direction,operation,CpuLauncher{},kernel,scheme,lap,dot);
    else execute(differential::Rows<2>{flat,buffers},t,support,direction,operation,CpuLauncher{},kernel,scheme,lap,dot);
}
}
Tensor corrected_jvp(const OptionalTensors& flat,const OptionalTensors& tangents,const Tensor& example,
        int64_t operation,int64_t support,int64_t direction,int64_t kernel,int64_t scheme,int64_t lap,bool dot,bool consistent) {
    STD_TORCH_CHECK(flat.size()==differential::Slots && flat[QX],"invalid derivative input layout");
    Tensor output;
    if (operation>=differential::Volume) {
        const auto& x=*flat[QX]; const auto d=x.size(1);
        output=stable::new_empty(x,{x.size(0),operation==differential::Moments ? 1+2*d+2*d*d+d*d*d:1});
    } else output=stable::new_empty(example,example.sizes());
    auto t=prepare_differential(flat,output,operation,support,direction,kernel,scheme,lap,dot,consistent);
    STD_TORCH_CHECK(tangents.size()==differential::Slots,"invalid tangent layout");
    for (int slot=0; slot<differential::Slots; ++slot) if (tangents[slot]) {
        STD_TORCH_CHECK(flat[slot],"Tangent requires a primal input");
        tensor(*tangents[slot],t[QX],t[QX].scalar_type(),"corrected tangent");
        STD_TORCH_CHECK(tangents[slot]->sizes()==flat[slot]->sizes(),"corrected tangent shape");
    }
    differential::Buffers buffers{tangents,{}, {},{},output,{},consistent};
    launch_differential(t,flat,buffers,operation,support,direction,kernel,scheme,lap,dot,1);
    return output;
}
OptionalTensors corrected_vjp(const OptionalTensors& flat,const Tensor& cotangent,const Tensors& transpose,
        int64_t mask,const std::optional<Tensor>& cache,int64_t operation,int64_t support,int64_t direction,
        int64_t kernel,int64_t scheme,int64_t lap,bool dot,bool consistent) {
    auto t=prepare_differential(flat,cotangent,operation,support,direction,kernel,scheme,lap,dot,consistent);
    OptionalTensors gradients(differential::Slots);
    bool reference=false;
    for (int slot=0; slot<differential::Slots; ++slot) if ((mask & (int64_t(1)<<slot)) && flat[slot]) {
        const auto& value=*flat[slot];
        auto shape=value.sizes().vec();
        if (slot==LO || slot==HI) shape={t[QX].size(0),t[QX].size(1)};
        gradients[slot]=stable::new_zeros(value,shape);
        // This predicate is independent of D.
        reference=reference || differential::Layout<1>::reference(slot);
    }
    if (reference) {
        STD_TORCH_CHECK(transpose.size()==3,"Reference pullback requires CSC");
        vector(transpose[0],t[QX],ScalarType::Long,t[RX].size(0),"CSC offsets");
        vector(transpose[1],t[QX],ScalarType::Int,t[RX].size(0),"CSC counts");
        vector(transpose[2],t[QX],ScalarType::Long,t[NEIGHBORS].numel(),"CSC IDs");
    }
    if (operation==differential::Volume || operation==differential::Consistency) {
        STD_TORCH_CHECK(cache,"Preparation pullback requires row accumulators");
        tensor(*cache,t[QX],t[QX].scalar_type(),"row accumulator cache");
        STD_TORCH_CHECK(cache->dim()==2 && cache->size(0)==t[QX].size(0) &&
            cache->size(1)==(operation==differential::Volume ? 1:2),"row accumulator cache shape");
    }
    const differential::Buffers buffers{{},gradients,transpose,cotangent,{},cache ? *cache:Tensor{},consistent};
    launch_differential(t,flat,buffers,operation,support,direction,kernel,scheme,lap,dot,2);
    return gradients;
}
// Cache only the row sums needed by nonlinear finalize pullbacks. This pass
// stores no per-edge values and is used by preparation, not core field maps.
Tensor preparation_accumulators(const OptionalTensors& flat,int64_t operation,int64_t support,int64_t direction,int64_t kernel) {
    STD_TORCH_CHECK(operation==differential::Volume || operation==differential::Consistency,"Only nonlinear preparation finalizers need accumulators");
    STD_TORCH_CHECK(flat.size()==differential::Slots && flat[QX],"invalid preparation input layout");
    const auto& x=*flat[QX];
    auto example=stable::new_empty(x,{x.size(0),1});
    auto t=prepare_differential(flat,example,operation,support,direction,kernel,1,2,false,false);
    auto cache=stable::new_empty(x,{x.size(0),operation==differential::Volume ? 1:2});
    t[OUTPUT]=cache;
    const differential::Buffers buffers{{},{},{},{},cache,{},false};
    launch_differential(t,flat,buffers,operation,support,direction,kernel,1,2,false,0);
    return cache;
}
STABLE_TORCH_LIBRARY_FRAGMENT(warpsph_native,m) {
    m.def("corrected_jvp(Tensor?[] flat, Tensor?[] tangents, Tensor example, int operation, int support, int direction, int kernel, int scheme, int lap, bool dot, bool consistent) -> Tensor");
    m.impl("corrected_jvp",TORCH_BOX(&corrected_jvp));
    m.def("corrected_vjp(Tensor?[] flat, Tensor cotangent, Tensor[] transpose, int mask, Tensor? cache, int operation, int support, int direction, int kernel, int scheme, int lap, bool dot, bool consistent) -> Tensor?[]");
    m.impl("corrected_vjp",TORCH_BOX(&corrected_vjp));
    m.def("preparation_accumulators(Tensor?[] flat, int operation, int support, int direction, int kernel) -> Tensor");
    m.impl("preparation_accumulators",TORCH_BOX(&preparation_accumulators));
}
} // namespace sph_native
