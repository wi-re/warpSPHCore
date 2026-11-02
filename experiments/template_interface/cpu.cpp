#include "dispatch.h"
#include "derivatives.h"
#include "execution/cpu.h"
#include <algorithm>

namespace prototype {

struct CpuForward {
    const Problem& problem;
    const Topology& topology;
    bool direct;

    template <class T, int D, int Fields, class Operation>
    void run(Operation operation) const {
        const ReferenceLoader<T,D,Fields,Operation::weight> loader{operation.fields};
        const CpuLauncher launch;
        if (topology.mode == 0) {
            if (topology.wide) {
                if (direct) launch(problem.queries, DirectRow<Operation,decltype(loader),int64_t>{operation,loader,csr<int64_t>(topology)});
                else launch(problem.queries, Row<Operation,Csr<int64_t>,decltype(loader)>{operation,csr<int64_t>(topology),loader});
            } else {
                if (direct) launch(problem.queries, DirectRow<Operation,decltype(loader),int32_t>{operation,loader,csr<int32_t>(topology)});
                else launch(problem.queries, Row<Operation,Csr<int32_t>,decltype(loader)>{operation,csr<int32_t>(topology),loader});
            }
        } else if (topology.mode == 1) {
            launch(problem.queries, Row<Operation,Ranges,decltype(loader)>{operation,ranges(topology),loader});
        } else {
            // CPU reference execution of the same query/source tile plan.
            torch::stable::parallel_for(0,topology.tasks,1,[&](int64_t begin,int64_t end) {
                for (int64_t task=begin;task<end;++task) {
                    for (int64_t i=topology.query_begin[task];i<topology.query_end[task];++i) {
                        const auto query=operation.prepare(i);
                        auto state=operation.identity(query);
                        if (operation.allowed(i)) ranges(topology).each(task,0,1,
                            Accumulate<Operation,decltype(query),decltype(loader),decltype(state)>{operation,query,loader,state});
                        operation.finish(query,state);
                    }
                }
            });
        }
    }
};

template <class T, int D, int Fields>
void zero_gradients(const Problem& a,const Derivatives& d) {
    const int64_t sizes[7]={int64_t(a.queries)*D,int64_t(a.references)*D,a.queries,a.references,
        a.references,int64_t(a.queries)*Fields,int64_t(a.references)*Fields};
    void* pointers[7]={d.gx,d.grx,d.gh,d.gm,d.grho,d.gqvalue,d.grvalue};
    for(int slot=0;slot<7;++slot) if(pointers[slot])
        std::fill_n(static_cast<T*>(pointers[slot]),sizes[slot],T(0));
}

struct CpuDerivative {
    const Problem& problem;
    const Topology &forward,&transpose;
    const Outputs& output;
    const Derivatives& derivatives;
    int mode;

    template <class T,int D,int Fields,bool Gradient,bool Normalize,bool Density,bool FieldsOnly>
    void execute() const {
        const auto fields=field_view<T,D,Fields>(problem,output);
        if(mode==1) {
            Tangent<T,D,Fields,Gradient,Normalize,Density,FieldsOnly> op{};
            op.fields=fields;op.derivatives=derivatives;
            using Loader=ReferenceLoader<T,D,Fields,decltype(op)::weight>;
            if(forward.wide) CpuLauncher{}(problem.queries,Row<decltype(op),Csr<int64_t>,Loader>{op,csr<int64_t>(forward),Loader{fields}});
            else CpuLauncher{}(problem.queries,Row<decltype(op),Csr<int32_t>,Loader>{op,csr<int32_t>(forward),Loader{fields}});
        } else {
            if constexpr (Normalize)
                CpuLauncher{}(problem.queries,FinalizationPullback<T,D>{
                    static_cast<const T*>(output.cache),static_cast<const T*>(derivatives.cotangent),
                    static_cast<T*>(derivatives.final_bars)});
            using PB=Pullback<T,D,Fields,Gradient,Normalize,Density,FieldsOnly>;
            const PB pb{fields,derivatives,static_cast<const T*>(output.cache)};
            if(mode==3) {
                zero_gradients<T,D,Fields>(problem,derivatives);
                if(forward.wide) CpuLauncher{}(problem.queries,AtomicAdjointRow<T,D,Fields,PB,Csr<int64_t>>{pb,csr<int64_t>(forward)});
                else CpuLauncher{}(problem.queries,AtomicAdjointRow<T,D,Fields,PB,Csr<int32_t>>{pb,csr<int32_t>(forward)});
            } else {
                if(derivatives.gx || derivatives.gh || derivatives.gqvalue) {
                    if(forward.wide) CpuLauncher{}(problem.queries,OwnedAdjointRow<Owner::Query,PB,Csr<int64_t>>{pb,csr<int64_t>(forward)});
                    else CpuLauncher{}(problem.queries,OwnedAdjointRow<Owner::Query,PB,Csr<int32_t>>{pb,csr<int32_t>(forward)});
                }
                if(derivatives.grx || derivatives.gm || derivatives.grho || derivatives.grvalue) {
                    if(transpose.wide) CpuLauncher{}(problem.references,OwnedAdjointRow<Owner::Reference,PB,Csr<int64_t>>{pb,csr<int64_t>(transpose)});
                    else CpuLauncher{}(problem.references,OwnedAdjointRow<Owner::Reference,PB,Csr<int32_t>>{pb,csr<int32_t>(transpose)});
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

extern "C" int template_forward_cpu(const Problem* problem,const Topology* topology,
                                    const Outputs* output,int direct,int iterations) {
    for(int iteration=0;iteration<iterations;++iteration)
        prototype::dispatch(*problem,*output,prototype::CpuForward{*problem,*topology,bool(direct)});
    return 0;
}

extern "C" int template_derivative_cpu(const Problem* problem,const Topology* forward,
    const Topology* transpose,const Outputs* output,const Derivatives* derivatives,int mode,int iterations) {
    for(int iteration=0;iteration<iterations;++iteration)
        prototype::dispatch_derivative(*problem,
            prototype::CpuDerivative{*problem,*forward,*transpose,*output,*derivatives,mode});
    return 0;
}
