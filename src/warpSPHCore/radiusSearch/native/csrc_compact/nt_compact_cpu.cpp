#include "nt_compact_backend.h"
#include <algorithm>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sph_native::compact {
namespace {
template <class Row> void parallel(const Context &context, int64_t rows, Row row) {
#ifdef _OPENMP
    const int workers = context.cpu_threads ? context.cpu_threads : omp_get_max_threads();
#pragma omp parallel for num_threads(workers) schedule(static) if (rows >= 32)
#endif
    for (int64_t i = 0; i < rows; ++i)
        row(i);
}
} // namespace
Summary cpu_summary(const Problem &problem, const Grid &grid, const int64_t *keys) {
    Summary summary{};
    numeric(problem, [&](auto scalar, auto dimension) {
        const auto args = arguments<decltype(scalar), decltype(dimension)::value>(problem);
        for (int64_t i = 0; i < std::max(problem.query.count, problem.reference.count); ++i)
            summary = combine(summary, inspect_row(args, problem.query.count, i, grid, keys));
    });
    return summary;
}
void cpu_build(const Context &context, const Problem &problem, Data &data) {
    std::memset(data.buckets, 0, size_t(data.grid.buckets) * sizeof(Bucket));
    if (!data.references)
        return;
    numeric(problem, [&](auto scalar, auto dimension) {
        const auto args = arguments<decltype(scalar), decltype(dimension)::value>(problem);
        parallel(context, data.references,
                 KeyRow<decltype(scalar), decltype(dimension)::value>{
                     args, data.grid, data.particles, data.reference_keys});
    });
    std::sort(data.particles, data.particles + data.references);
    for (int64_t i = 0; i < data.references; ++i)
        if (i == 0 || data.particles[i].bucket != data.particles[i - 1].bucket ||
            data.particles[i].cell != data.particles[i - 1].cell)
            ++data.occupied;
    data.cells = static_cast<Cell *>(data.allocate(size_t(data.occupied) * sizeof(Cell)));
    int64_t slot = 0;
    for (int64_t begin = 0; begin < data.references;) {
        int64_t end = begin + 1;
        const auto [bucket, key, original] = data.particles[begin];
        while (end < data.references && data.particles[end].bucket == bucket &&
               data.particles[end].cell == key)
            ++end;
        data.cells[slot++] = {key, begin, end, bucket};
        begin = end;
    }
    parallel(context, data.occupied,
             FinishCellRow{data.cells, data.buckets, data.occupied, data.references});
}
void cpu_search(const Context &context, const Problem &problem, const Data &data, Buffer output,
                bool collect, int32_t *error) {
    numeric(problem, [&](auto scalar, auto dimension) {
        auto args = arguments<decltype(scalar), decltype(dimension)::value>(problem);
        if (collect)
            args.collected = static_cast<int64_t *>(output.data);
        else
            args.counts = static_cast<int32_t *>(output.data);
        parallel(context, problem.query.count,
                 SearchRow<decltype(scalar), decltype(dimension)::value>{args, data.view(), collect,
                                                                         error});
    });
}
} // namespace sph_native::compact
