#include "dispatch.h"
#include "derivatives.h"
#include "execution/neighborhood.cuh"

namespace prototype {

template <class Row>
void cooperative(int64_t size,Row row,int width,NativeStream stream) {
    if(width==8) CooperativeGpuLauncher<8>{stream}(size,CooperativeRow<8,Row>{row});
    else if(width==16) CooperativeGpuLauncher<16>{stream}(size,CooperativeRow<16,Row>{row});
    else CooperativeGpuLauncher<32>{stream}(size,CooperativeRow<32,Row>{row});
}

template <class Loader>
__global__ void pack_references(int64_t count,Loader loader,void* destination) {
    const int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(j<count) {
        auto record=loader.load(j);
        using Record=decltype(record);
        static_cast<Record*>(destination)[j]=record;
    }
}

struct GpuForward {
    const Problem& problem;
    const Topology& topology;
    const Outputs& output;
    int width;
    bool direct,pack;

    template <class T,int D,int Fields,class Operation>
    void run(Operation op) const {
        const auto stream=static_cast<NativeStream>(problem.stream);
        using RawLoader=ReferenceLoader<T,D,Fields,Operation::weight>;
        using Loader=ReferenceLoader<T,D,Fields,Operation::weight,true>;
        if(pack && problem.references)
            pack_references<<<(problem.references+255)/256,256,0,stream>>>(problem.references,RawLoader{op.fields},output.packed);
        const Loader loader{op.fields,static_cast<const T*>(output.packed)};
        if(topology.mode==0) {
            if(topology.wide) {
                if(direct) cooperative(problem.queries,DirectRow<Operation,Loader,int64_t>{op,loader,csr<int64_t>(topology)},width,stream);
                else cooperative(problem.queries,Row<Operation,Csr<int64_t>,Loader>{op,csr<int64_t>(topology),loader},width,stream);
            } else {
                if(direct) cooperative(problem.queries,DirectRow<Operation,Loader,int32_t>{op,loader,csr<int32_t>(topology)},width,stream);
                else cooperative(problem.queries,Row<Operation,Csr<int32_t>,Loader>{op,csr<int32_t>(topology),loader},width,stream);
            }
        } else if(topology.mode==1) {
            cooperative(problem.queries,Row<Operation,Ranges,Loader>{op,ranges(topology),loader},width,stream);
        } else if(topology.tasks) {
            // A shared tile owns up to eight queries and stages 64 references.
            // The operation callable is unchanged from CSR and CPU execution.
            shared_ranges<16,8,64><<<topology.tasks,128,0,stream>>>(op,loader,
                topology.query_begin,topology.query_end,ranges(topology));
        }
    }
};

template <class T,int D,int Fields>
void zero_gradients(const Problem& a,const Derivatives& d,NativeStream stream) {
    const int64_t sizes[7]={int64_t(a.queries)*D,int64_t(a.references)*D,a.queries,a.references,
        a.references,int64_t(a.queries)*Fields,int64_t(a.references)*Fields};
    void* pointers[7]={d.gx,d.grx,d.gh,d.gm,d.grho,d.gqvalue,d.grvalue};
    for(int slot=0;slot<7;++slot) if(pointers[slot] && sizes[slot])
        (void)GPU(MemsetAsync)(pointers[slot],0,sizes[slot]*sizeof(T),stream);
}

struct GpuDerivative {
    const Problem& problem;
    const Topology &forward,&transpose;
    const Outputs& output;
    const Derivatives& derivatives;
    int mode,width;

    template <class T,int D,int Fields,bool Gradient,bool Normalize,bool Density,bool FieldsOnly>
    void execute() const {
        const auto stream=static_cast<NativeStream>(problem.stream);
        const auto fields=field_view<T,D,Fields>(problem,output);
        if(mode==1) {
            Tangent<T,D,Fields,Gradient,Normalize,Density,FieldsOnly> op{};
            op.fields=fields;op.derivatives=derivatives;
            using Loader=ReferenceLoader<T,D,Fields,decltype(op)::weight>;
            if(forward.wide) cooperative(problem.queries,Row<decltype(op),Csr<int64_t>,Loader>{op,csr<int64_t>(forward),Loader{fields}},width,stream);
            else cooperative(problem.queries,Row<decltype(op),Csr<int32_t>,Loader>{op,csr<int32_t>(forward),Loader{fields}},width,stream);
        } else {
            if constexpr (Normalize)
                launch_pointwise(problem.queries,FinalizationPullback<T,D>{
                    static_cast<const T*>(output.cache),static_cast<const T*>(derivatives.cotangent),
                    static_cast<T*>(derivatives.final_bars)},stream);
            using PB=Pullback<T,D,Fields,Gradient,Normalize,Density,FieldsOnly>;
            const PB pb{fields,derivatives,static_cast<const T*>(output.cache)};
            if(mode==3) {
                zero_gradients<T,D,Fields>(problem,derivatives,stream);
                if(forward.wide) cooperative(problem.queries,AtomicAdjointRow<T,D,Fields,PB,Csr<int64_t>>{pb,csr<int64_t>(forward)},width,stream);
                else cooperative(problem.queries,AtomicAdjointRow<T,D,Fields,PB,Csr<int32_t>>{pb,csr<int32_t>(forward)},width,stream);
            } else {
                if(derivatives.gx || derivatives.gh || derivatives.gqvalue) {
                    if(forward.wide) cooperative(problem.queries,OwnedAdjointRow<Owner::Query,PB,Csr<int64_t>>{pb,csr<int64_t>(forward)},width,stream);
                    else cooperative(problem.queries,OwnedAdjointRow<Owner::Query,PB,Csr<int32_t>>{pb,csr<int32_t>(forward)},width,stream);
                }
                if(derivatives.grx || derivatives.gm || derivatives.grho || derivatives.grvalue) {
                    if(transpose.wide) cooperative(problem.references,OwnedAdjointRow<Owner::Reference,PB,Csr<int64_t>>{pb,csr<int64_t>(transpose)},width,stream);
                    else cooperative(problem.references,OwnedAdjointRow<Owner::Reference,PB,Csr<int32_t>>{pb,csr<int32_t>(transpose)},width,stream);
                }
            }
        }
    }

    template <class T,int D,int Fields,bool Gradient,bool Normalize,bool Density>
    void run() const {
        const bool fields_only = mode==1 ?
            !(derivatives.tx || derivatives.trx || derivatives.th || derivatives.tm || derivatives.trho) :
            !(derivatives.gx || derivatives.grx || derivatives.gh || derivatives.gm || derivatives.grho);
        if constexpr (Fields>0) {
            if(fields_only) {
                execute<T,D,Fields,Gradient,Normalize,Density,true>();
                return;
            }
        }
        execute<T,D,Fields,Gradient,Normalize,Density,false>();
    }
};
} // namespace prototype

extern "C" int template_forward_gpu(const Problem* problem,const Topology* topology,
    const Outputs* output,int width,int direct,int pack,int iterations) {
    for(int iteration=0;iteration<iterations;++iteration)
        prototype::dispatch(*problem,*output,
            prototype::GpuForward{*problem,*topology,*output,width,bool(direct),bool(pack)});
    return int(GPU(GetLastError)());
}

extern "C" int template_derivative_gpu(const Problem* problem,const Topology* forward,
    const Topology* transpose,const Outputs* output,const Derivatives* derivatives,
    int mode,int width,int iterations) {
    for(int iteration=0;iteration<iterations;++iteration)
        prototype::dispatch_derivative(*problem,
            prototype::GpuDerivative{*problem,*forward,*transpose,*output,*derivatives,mode,width});
    return int(GPU(GetLastError)());
}
