#include "nt_compact_backend.h"
#include "execution/device_runtime.cuh"
#include "execution/scan.h"
#include "execution/rowSort.cuh"
#include <algorithm>
#include <stdexcept>
#if !defined(__HIPCC__) && !defined(SPH_NATIVE_HIP) && !defined(USE_ROCM)
#include <cub/device/device_merge_sort.cuh>
#endif

namespace sph_native::compact {
using runtime::check;
using runtime::Guard;
namespace {
constexpr int block_size = 256;
struct ParticleLess {
    SPH_HD bool operator()(const ParticleKey &a, const ParticleKey &b) const { return a < b; }
};
template <class Row> __global__ void rows(int64_t count, Row row) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x)
        row(i);
}
template <class Row> void launch(int64_t count, Row row, NativeStream stream) {
    if (!count)
        return;
    const int blocks = int(std::min<int64_t>((count - 1) / block_size + 1, 65535));
    rows<<<blocks, block_size, 0, stream>>>(count, row);
    check(GPU(GetLastError)());
}
__device__ Summary block_summary(Summary value) {
    __shared__ double query[block_size], reference[block_size];
    const int lane = threadIdx.x;
    query[lane] = value.query_max;
    reference[lane] = value.reference_max;
    __syncthreads();
    for (int stride = block_size / 2; stride; stride /= 2) {
        if (lane < stride) {
            query[lane] = query[lane] > query[lane + stride] ? query[lane] : query[lane + stride];
            reference[lane] = reference[lane] > reference[lane + stride] ? reference[lane]
                                                                         : reference[lane + stride];
        }
        __syncthreads();
    }
    return {query[0], reference[0]};
}
template <class T, int D>
__global__ void summarize_inputs(RowArgs<T, D> args, int64_t queries, Summary *partials) {
    Summary value{};
    const int64_t count = queries > args.references ? queries : args.references;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x)
        value = combine(value, support_bounds(args, queries, i));
    const auto result = block_summary(value);
    if (!threadIdx.x)
        partials[blockIdx.x] = result;
}
__global__ void summarize_partials(const Summary *partials, int count, Summary *output) {
    const auto result = block_summary(threadIdx.x < count ? partials[threadIdx.x] : Summary{});
    if (!threadIdx.x)
        *output = result;
}
} // namespace

void *gpu_allocate(const Context &context, size_t bytes) {
    Guard guard(context.device_index);
    void *allocation = nullptr;
    check(GPU(Malloc)(&allocation, bytes));
    return allocation;
}
void gpu_free(const Context &context, void *allocation) noexcept {
    int previous = 0;
    if (GPU(GetDevice)(&previous) != GPU(Success))
        return;
    if (GPU(SetDevice)(context.device_index) == GPU(Success))
        (void)GPU(Free)(allocation);
    (void)GPU(SetDevice)(previous);
}
void gpu_domain(const Context &context, const Problem &problem, double *lower, double *upper,
                bool *periodic) {
    Guard guard(context.device_index);
    const auto stream = static_cast<NativeStream>(context.stream);
    numeric(problem, [&](auto scalar, auto dimension) {
        using T = decltype(scalar);
        constexpr int D = decltype(dimension)::value;
        T lo[D], hi[D];
        uint8_t flags[D];
        check(GPU(MemcpyAsync)(lo, problem.domain.min.data, sizeof(lo), GPU(MemcpyDeviceToHost),
                               stream));
        check(GPU(MemcpyAsync)(hi, problem.domain.max.data, sizeof(hi), GPU(MemcpyDeviceToHost),
                               stream));
        check(GPU(MemcpyAsync)(flags, problem.domain.periodic.data, sizeof(flags),
                               GPU(MemcpyDeviceToHost), stream));
        check(GPU(StreamSynchronize)(stream));
        for (int axis = 0; axis < D; ++axis) {
            if (flags[axis] > 1)
                throw std::invalid_argument("Invalid domain periodicity");
            lower[axis] = lo[axis];
            upper[axis] = hi[axis];
            periodic[axis] = bool(flags[axis]);
        }
    });
}
Summary gpu_summary(const Context &context, const Problem &problem) {
    Guard guard(context.device_index);
    const auto stream = static_cast<NativeStream>(context.stream);
    const auto count = std::max(problem.query.count, problem.reference.count);
    if (!count)
        return {};
    const int blocks = int(std::min<int64_t>((count - 1) / block_size + 1, block_size));
    Data scratch{};
    scratch.context = context;
    auto *partials = static_cast<Summary *>(scratch.allocate(size_t(blocks) * sizeof(Summary)));
    auto *result = static_cast<Summary *>(scratch.allocate(sizeof(Summary)));
    numeric(problem, [&](auto scalar, auto dimension) {
        const auto args = arguments<decltype(scalar), decltype(dimension)::value>(problem);
        summarize_inputs<<<blocks, block_size, 0, stream>>>(args, problem.query.count, partials);
        check(GPU(GetLastError)());
    });
    summarize_partials<<<1, block_size, 0, stream>>>(partials, blocks, result);
    check(GPU(GetLastError)());
    Summary host{};
    check(GPU(MemcpyAsync)(&host, result, sizeof(host), GPU(MemcpyDeviceToHost), stream));
    check(GPU(StreamSynchronize)(stream));
    return host;
}
void gpu_build(const Context &context, const Problem &problem, Data &data) {
    Guard guard(context.device_index);
    const auto stream = static_cast<NativeStream>(context.stream);
    check(GPU(MemsetAsync)(data.buckets, 0, size_t(data.grid.buckets) * sizeof(Bucket), stream));
    Data scratch{};
    scratch.context = context;
    if (data.references) {
        numeric(problem, [&](auto scalar, auto dimension) {
            const auto args = arguments<decltype(scalar), decltype(dimension)::value>(problem);
            launch(data.references,
                   KeyRow<decltype(scalar), decltype(dimension)::value>{args, data.grid,
                                                                        data.particles},
                   stream);
        });
#if !defined(__HIPCC__) && !defined(SPH_NATIVE_HIP) && !defined(USE_ROCM)
        size_t temporary_bytes = 0;
        check(cub::DeviceMergeSort::SortKeys(nullptr, temporary_bytes, data.particles,
                                             data.references, ParticleLess{}, stream));
        void *temporary = scratch.allocate(std::max<size_t>(1, temporary_bytes));
        check(cub::DeviceMergeSort::SortKeys(temporary, temporary_bytes, data.particles,
                                             data.references, ParticleLess{}, stream));
#else
        auto *temporary = static_cast<ParticleKey *>(
            scratch.allocate(size_t(data.references) * sizeof(ParticleKey)));
        auto *input = data.particles, *output = temporary;
        for (int64_t run = 1; run < data.references; run *= 2) {
            launch(data.references, MergeRow{input, output, data.references, run}, stream);
            std::swap(input, output);
        }
        if (input != data.particles)
            check(GPU(MemcpyAsync)(data.particles, input,
                                   size_t(data.references) * sizeof(ParticleKey),
                                   GPU(MemcpyDeviceToDevice), stream));
#endif
        auto *heads =
            static_cast<int32_t *>(scratch.allocate(size_t(data.references) * sizeof(int32_t)));
        auto *offsets =
            static_cast<int64_t *>(scratch.allocate(size_t(data.references) * sizeof(int64_t)));
        launch(data.references, BoundaryRow{data.particles, heads}, stream);
        data.occupied = runtime::gpu_scan(context, data.references, {heads, data.references},
                                          {offsets, data.references});
        data.cells = static_cast<Cell *>(data.allocate(size_t(data.occupied) * sizeof(Cell)));
        launch(data.references, EmitCellRow{data.particles, heads, offsets, data.cells}, stream);
        launch(data.occupied,
               FinishCellRow{data.cells, data.buckets, data.occupied, data.references}, stream);
    }
    // Construction owns scratch buffers and publishes a fully built index.
    check(GPU(StreamSynchronize)(stream));
}
void gpu_search(const Context &context, const Problem &problem, const Data &data, Buffer output,
                bool collect) {
    Guard guard(context.device_index);
    const auto stream = static_cast<NativeStream>(context.stream);
    numeric(problem, [&](auto scalar, auto dimension) {
        auto args = arguments<decltype(scalar), decltype(dimension)::value>(problem);
        if (collect) {
            args.collected = static_cast<int64_t *>(output.data);
            args.neighbors = {output.data, true};
        } else
            args.counts = static_cast<int32_t *>(output.data);
        launch(problem.query.count,
               SearchRow<decltype(scalar), decltype(dimension)::value, false>{args, data.view(),
                                                                              collect},
               stream);
        if (collect)
            sort_csr_rows(problem.query.count, args.offsets, args.counts, args.collected, stream);
    });
}
} // namespace sph_native::compact
