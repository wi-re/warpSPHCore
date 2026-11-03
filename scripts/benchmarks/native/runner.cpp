// Timing-only C boundary. Fixtures are borrowed, but every timed iteration runs
// entirely in C++ and calls the public framework-independent API.
#include "api.h"
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef BENCH_CUDA
#include <cuda_runtime_api.h>
#elif defined(BENCH_HIP)
#include <hip/hip_runtime.h>
#endif

namespace {
using namespace sph_native;
// Keep this layout in sync with NativeCase in bench_backends.py.
struct Case {
    int device, dimension, operation, threads, precision;
    int64_t particles, edges;
    void *stream;
    const void *positions, *supports, *masses, *densities, *values, *lower, *upper;
    const bool *periodic;
    const int64_t *offsets, *indices;
    const int32_t *counts, *kinds;
    void *output;
};
thread_local std::string error;

void synchronize(const Context &context) {
#ifdef BENCH_CUDA
    if (context.device == Device::CUDA &&
        cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)) != cudaSuccess)
        throw std::runtime_error("CUDA benchmark synchronization failed");
#elif defined(BENCH_HIP)
    if (context.device == Device::HIP &&
        hipStreamSynchronize(static_cast<hipStream_t>(context.stream)) != hipSuccess)
        throw std::runtime_error("HIP benchmark synchronization failed");
#endif
}

// Search owns fresh count, offset and edge arrays on every iteration. Operator
// mode can either allocate its output or reuse the caller's preallocated array.
class Storage {
    void *pointer_ = nullptr;
    Device device_;

  public:
    Storage(Device device, size_t bytes) : device_(device) {
        bytes = bytes ? bytes : 1;
        if (device == Device::CPU) {
            pointer_ = std::malloc(bytes);
            if (!pointer_)
                throw std::bad_alloc();
            return;
        }
#ifdef BENCH_CUDA
        if (device == Device::CUDA && cudaMalloc(&pointer_, bytes) == cudaSuccess)
            return;
#elif defined(BENCH_HIP)
        if (device == Device::HIP && hipMalloc(&pointer_, bytes) == hipSuccess)
            return;
#endif
        throw std::runtime_error("Device allocation failed or unsupported benchmark device");
    }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    ~Storage() {
        if (device_ == Device::CPU)
            std::free(pointer_);
#ifdef BENCH_CUDA
        if (device_ == Device::CUDA)
            cudaFree(pointer_);
#elif defined(BENCH_HIP)
        if (device_ == Device::HIP)
            (void)hipFree(pointer_);
#endif
    }
    void *data() const { return pointer_; }
};

Problem problem(const Case &c) {
    const auto n = c.particles;
    const auto d = c.dimension;
    if (c.precision != int(Precision::Float32) && c.precision != int(Precision::Float64))
        throw std::invalid_argument("Unsupported benchmark precision");
    Problem p{};
    p.precision = Precision(c.precision);
    p.dimension = d;
    p.field_rank = 1;
    p.query = {n,           {c.positions, n * d}, {c.supports, n}, {c.masses, n}, {c.densities, n},
               {c.kinds, n}};
    p.reference = p.query;
    p.domain = {{c.lower, d}, {c.upper, d}, {c.periodic, d}};
    p.adjacency = {{c.offsets, n}, {c.counts, n}, {c.indices, c.edges}, 64, 64};
    p.query_values = p.reference_values = {c.values, n * d};
    p.properties = {Operation(c.operation),
                    Kernel::Wendland2,
                    Support::Gather,
                    Scheme::Difference,
                    Laplacian::Brookshaw,
                    0,
                    false};
    return p;
}

void search(const Context &context, Problem p, Buffer collected = {}) {
    const auto n = p.query.count;
    auto index = build_compact_hash(context, p);
    Storage counts(context.device, n * sizeof(int32_t));
    Storage offsets(context.device, n * sizeof(int64_t));
    radius_count(context, p, index, {counts.data(), n});
    const auto edges = radius_scan(context, n, {counts.data(), n}, {offsets.data(), n});
    p.adjacency = {{offsets.data(), n}, {counts.data(), n}, {}, 64, 64};
    if (collected.data) {
        if (edges != collected.size)
            throw std::runtime_error("Search edge count mismatch");
        radius_collect(context, p, index, collected);
    } else {
        Storage indices(context.device, edges * sizeof(int64_t));
        radius_collect(context, p, index, {indices.data(), edges});
    }
}
} // namespace

extern "C" int native_benchmark_abi_version() { return 2; }
extern "C" const char *native_benchmark_error() { return error.c_str(); }
// task 0: allocating forward; 1: full allocating search;
// task 2: forward into retained output.
// completed_ms includes stream completion. submitted_ms excludes only the final
// synchronization; API-internal synchronization remains part of both metrics.
extern "C" int native_benchmark(const Case *fixture, int task, int iterations, double *completed_ms,
                                double *submitted_ms) {
    try {
        error.clear();
        if (iterations < 1 || task < 0 || task > 2)
            throw std::invalid_argument("Invalid timing request");
        const auto &c = *fixture;
        const Context context{Device(c.device), 0, c.stream, c.threads};
        const auto p = problem(c);
        const auto elements = output_elements(p);
        const auto scalar_bytes = p.precision == Precision::Float32 ? sizeof(float) : sizeof(double);
        synchronize(context);
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (task == 1)
                search(context, p);
            else if (task == 0) {
                Storage output(context.device, elements * scalar_bytes);
                forward(context, p, {output.data(), elements});
            } else
                forward(context, p, {c.output, elements});
        }
        const auto submitted = std::chrono::steady_clock::now();
        synchronize(context);
        const auto completed = std::chrono::steady_clock::now();
        *completed_ms =
            std::chrono::duration<double, std::milli>(completed - begin).count() / iterations;
        *submitted_ms =
            std::chrono::duration<double, std::milli>(submitted - begin).count() / iterations;
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return 1;
    }
}

// Untimed correctness check using precisely the same search implementation as
// the timed loop. The caller provides an edge array for exact CSR comparison.
extern "C" int native_benchmark_search(const Case *fixture, int64_t *indices) {
    try {
        error.clear();
        const auto &c = *fixture;
        const Context context{Device(c.device), 0, c.stream, c.threads};
        search(context, problem(c), {indices, c.edges});
        synchronize(context);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return 1;
    }
}
