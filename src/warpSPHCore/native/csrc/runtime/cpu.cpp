#include "runtime/backend.h"
#include "coreOperations/native/csrc_dispatch/nt_operations.h"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace sph_native::runtime {
struct ParallelCPU {
    int threads;
    template <class Row> void operator()(int64_t n, Row row) const {
#ifdef _OPENMP
        int workers = threads ? threads : omp_get_max_threads();
#pragma omp parallel for num_threads(workers) schedule(static) if (n >= 32)
#endif
        for (int64_t i = 0; i < n; ++i)
            row(i);
    }
};
void cpu_run(Task task, const Context &ctx, const Prepared &p, Buffer out, ConstBuffer cot,
             const Adjoints &bars) {
    int32_t error = 0;
    numeric(p, [&](auto scalar, auto dimension) {
        using T = decltype(scalar);
        constexpr int D = decltype(dimension)::value;
        run_rows(task, p, row_args<T, D>(p, out), out, cot, bars, &error,
                 ParallelCPU{ctx.cpu_threads});
    });
    if (error)
        throw std::invalid_argument(
            "Radius counts no longer match the geometry; recount before collection");
}
} // namespace sph_native::runtime
