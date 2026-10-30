#include "runtime/backend.h"
#include <algorithm>
#include <string>
#include "runtime/gpu_runtime.cuh"
#include "coreOperations/native/csrc_dispatch/nt_operations.h"
namespace sph_native::runtime {
template <class Row> __global__ void rows(int64_t n, Row row) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += int64_t(blockDim.x) * gridDim.x)
        row(i);
}
struct GPUExecute {
    NativeStream stream;
    template <class Row> void operator()(int64_t n, Row row) const {
        if (!n)
            return;
        rows<<<int(std::min<int64_t>((n + 127) / 128, 65535)), 128, 0, stream>>>(n, row);
        check(GPU(GetLastError)());
    }
};
struct ErrorBuffer {
    int32_t *device = nullptr;
    ErrorBuffer() { check(GPU(Malloc)(reinterpret_cast<void **>(&device), sizeof(int32_t))); }
    ~ErrorBuffer() {
        if (device)
            (void)GPU(Free)(device);
    }
};
void gpu_run(Task task, const Context &c, const Prepared &p, Buffer out, ConstBuffer cot,
             const Adjoints &bars) {
    Guard guard(c.device_index);
    auto stream = static_cast<NativeStream>(c.stream);
    // Only count/collect needs a host-visible completion status. Primal and VJP
    // calls enqueue work on the caller's stream without a final synchronize.
    if (task == Task::Collect) {
        ErrorBuffer error;
        check(GPU(MemsetAsync)(error.device, 0, sizeof(int32_t), stream));
        numeric(p, [&](auto scalar, auto dim) {
            using T = decltype(scalar);
            constexpr int D = decltype(dim)::value;
            run_rows(task, p, row_args<T, D>(p, out), out, cot, bars, error.device,
                     GPUExecute{stream});
        });
        int32_t host = 0;
        check(GPU(MemcpyAsync)(&host, error.device, sizeof(host), GPU(MemcpyDeviceToHost), stream));
        check(GPU(StreamSynchronize)(stream));
        if (host)
            throw std::invalid_argument(
                "Radius counts no longer match the geometry; recount before collection");
    } else
        numeric(p, [&](auto scalar, auto dim) {
            using T = decltype(scalar);
            constexpr int D = decltype(dim)::value;
            run_rows(task, p, row_args<T, D>(p, out), out, cot, bars, nullptr, GPUExecute{stream});
        });
}
void gpu_read(const Context &c, void *dest, const void *src, size_t n) {
    Guard guard(c.device_index);
    auto stream = static_cast<NativeStream>(c.stream);
    check(GPU(MemcpyAsync)(dest, src, n, GPU(MemcpyDeviceToHost), stream));
    check(GPU(StreamSynchronize)(stream));
}
void gpu_write(const Context &c, void *dest, const void *src, size_t n) {
    Guard guard(c.device_index);
    auto stream = static_cast<NativeStream>(c.stream);
    check(GPU(MemcpyAsync)(dest, src, n, GPU(MemcpyHostToDevice), stream));
    check(GPU(StreamSynchronize)(stream));
}
void gpu_zero(const Context &c, void *dest, size_t n) {
    Guard guard(c.device_index);
    check(GPU(MemsetAsync)(dest, 0, n, static_cast<NativeStream>(c.stream)));
}
} // namespace sph_native::runtime
