#pragma once
#include "api.h"
#include "operators.h"

namespace prototype {

template <class T, int D, int Fields>
SphFields<T, D, Fields> field_view(const Problem& a, const Outputs& out) {
    SphFields<T, D, Fields> fields{};
    fields.query_positions = static_cast<const T*>(a.qx);
    fields.reference_positions = static_cast<const T*>(a.rx);
    fields.supports = static_cast<const T*>(a.h);
    fields.mass = static_cast<const T*>(a.mass);
    fields.density = static_cast<const T*>(a.density);
    fields.query_values = static_cast<const T*>(a.qvalue);
    fields.reference_values = static_cast<const T*>(a.rvalue);
    fields.volumes = static_cast<const T*>(a.volume);
    fields.query_kinds = a.qkind; fields.reference_kinds = a.rkind;
    fields.lower = static_cast<const T*>(a.lower); fields.upper = static_cast<const T*>(a.upper);
    fields.periodic = a.periodic; fields.direction = a.direction;
    fields.output = static_cast<T*>(out.values); fields.cache = static_cast<T*>(out.cache);
    fields.counts = out.counts; fields.winners = out.winners;
    return fields;
}

template <class T, int D, class Task> void operation(const Problem& a, const Outputs& out, Task task) {
#define UNPAREN(...) __VA_ARGS__
#define RUN(TYPE,F) { UNPAREN TYPE op{}; op.fields = field_view<T,D,F>(a,out); task.template run<T,D,F>(op); }
    switch (a.work) {
    case 0: RUN((Density<T,D>),0); break;
    case 1: RUN((Interpolate<T,D,1>),1); break;
    case 2: RUN((Gradient<T,D,1>),1); break;
    case 3: RUN((Gradient<T,D,D>),D); break;
    case 4: RUN((Covariance<T,D>),0); break;
    case 5: RUN((Interpolate<T,D,1,true>),1); break;
    case 6: RUN((Moments<T,D>),0); break;
    case 7: RUN((MaxField<T,D>),1); break;
    case 8: RUN((Interpolate<T,D,D>),D); break;
    }
#undef RUN
#undef UNPAREN
}

template <class Task> void dispatch(const Problem& a, const Outputs& out, Task task) {
#define DIM(T) switch(a.dimension) { case 1: operation<T,1>(a,out,task); break; case 2: operation<T,2>(a,out,task); break; case 3: operation<T,3>(a,out,task); break; }
    if (a.precision) { DIM(double); } else { DIM(float); }
#undef DIM
}

template <class T, int D, class Task> void derivative_operation(const Problem& a, Task task) {
    switch(a.work) {
    case 0: task.template run<T,D,0,false,false,true>(); break;
    case 1: task.template run<T,D,1,false,false,false>(); break;
    case 2: task.template run<T,D,1,true,false,false>(); break;
    case 3: task.template run<T,D,D,true,false,false>(); break;
    case 5: task.template run<T,D,1,false,true,false>(); break;
    case 8: task.template run<T,D,D,false,false,false>(); break;
    }
}
template <class Task> void dispatch_derivative(const Problem& a, Task task) {
#define DIM(T) switch(a.dimension) { case 1: derivative_operation<T,1>(a,task); break; case 2: derivative_operation<T,2>(a,task); break; case 3: derivative_operation<T,3>(a,task); break; }
    if (a.precision) { DIM(double); } else { DIM(float); }
#undef DIM
}

// Direct-loop control: same prepared records, interaction and reduction as the
// template row, with neighborhood orchestration written out explicitly.
template <class Operation, class Loader, class Index> struct DirectRow {
    Operation operation;
    Loader loader;
    Csr<Index> csr;
    template <class Reduction>
    SPH_HD void run(int64_t i, int lane, int lanes, const Reduction& reduction) const {
        const auto query = operation.prepare(i);
        auto state = operation.identity(query);
        if (operation.allowed(i)) {
            for (int edge = lane; edge < csr.counts[i]; edge += lanes) {
                const auto at = csr.offsets[i] + edge;
                const auto j = int64_t(csr.indices[at]);
                if (loader.allowed(j))
                    operation.interact(query, loader.load(j), Neighbor{j, at}, state);
            }
        }
        state.reduce(reduction);
        if (lane == 0) operation.finish(query, state);
    }
    SPH_HD void operator()(int64_t i) const { run(i,0,1,LocalReduction{}); }
};

inline Ranges ranges(const Topology& topology) {
    return {topology.offsets, topology.counts, topology.begins, topology.ends};
}
template <class Index> Csr<Index> csr(const Topology& topology) {
    return {topology.offsets, topology.counts, static_cast<const Index*>(topology.indices)};
}
} // namespace prototype
