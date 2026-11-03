#pragma once
#include "radiusSearch/native/csrc_radius/nt_radius_forward.h"
#include <vector>
#include <type_traits>
#include <stdexcept>

#if defined(SPH_NATIVE_HAS_GPU) || defined(WITH_CUDA) || defined(__CUDACC__) || defined(__HIPCC__)
#define SPH_COMPACT_HAS_GPU
#endif
namespace sph_native::compact {
struct Data {
    Context context;
    Precision precision;
    int dimension;
    int64_t references, occupied = 0;
    Grid grid;
    double radius_bound = 0;
    ParticleKey *particles = nullptr;
    Cell *cells = nullptr;
    Bucket *buckets = nullptr;
    std::vector<void *> allocations;
    ~Data();
    void *allocate(size_t bytes);
    View view() const { return {grid, particles, cells, buckets, occupied}; }
};
struct Access {
    static Data &get(const CompactHash &);
};
void validate_geometry(const Context &, const Problem &);
void validate_index(const Context &, const Problem &, const Data &);
View view(const CompactHash &);
Summary cpu_summary(const Problem &);
void cpu_build(const Context &, const Problem &, Data &);
void cpu_search(const Context &, const Problem &, const Data &, Buffer, bool);
#ifdef SPH_COMPACT_HAS_GPU
Summary gpu_summary(const Context &, const Problem &);
void gpu_domain(const Context &, const Problem &, double *, double *, bool *);
void *gpu_allocate(const Context &, size_t);
void gpu_free(const Context &, void *) noexcept;
void gpu_build(const Context &, const Problem &, Data &);
void gpu_search(const Context &, const Problem &, const Data &, Buffer, bool);
#endif
void search(const Context &, const Problem &, const CompactHash &, Buffer, bool);

template <class F> void numeric(const Problem &problem, const F &function) {
    auto dimensions = [&](auto scalar) {
        if (problem.dimension == 1)
            function(scalar, std::integral_constant<int, 1>{});
        else if (problem.dimension == 2)
            function(scalar, std::integral_constant<int, 2>{});
        else
            function(scalar, std::integral_constant<int, 3>{});
    };
    if (problem.precision == Precision::Float32)
        dimensions(float{});
    else
        dimensions(double{});
}
template <class T, int D> RowArgs<T, D> arguments(const Problem &problem) {
    RowArgs<T, D> args{};
    args.qx = static_cast<const T *>(problem.query.positions.data);
    args.rx = static_cast<const T *>(problem.reference.positions.data);
    args.qh = static_cast<const T *>(problem.query.supports.data);
    args.rh = static_cast<const T *>(problem.reference.supports.data);
    args.lo = static_cast<const T *>(problem.domain.min.data);
    args.hi = static_cast<const T *>(problem.domain.max.data);
    args.periodic = static_cast<const bool *>(problem.domain.periodic.data);
    args.references = problem.reference.count;
    args.support = int(problem.properties.support);
    args.counts =
        const_cast<int32_t *>(static_cast<const int32_t *>(problem.adjacency.counts.data));
    args.offsets = {problem.adjacency.offsets.data, problem.adjacency.offsets_bits == 64};
    return args;
}
} // namespace sph_native::compact
