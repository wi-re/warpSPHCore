#include "runtime/backend.h"
#include <algorithm>
#include <string>
#include "execution/device_runtime.cuh"
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
void gpu_run(Task task, const Context &c, const Prepared &p, Buffer out, ConstBuffer cot,
             const Adjoints &bars) {
    Guard guard(c.device_index);
    auto stream = static_cast<NativeStream>(c.stream);
    numeric(p, [&](auto scalar, auto dim) {
        using T = decltype(scalar);
        constexpr int D = decltype(dim)::value;
        run_rows(task, p, row_args<T, D>(p, out), out, cot, bars, GPUExecute{stream});
    });
}

void gpu_zero(const Context &c, void *dest, size_t n) {
    Guard guard(c.device_index);
    check(GPU(MemsetAsync)(dest, 0, n, static_cast<NativeStream>(c.stream)));
}
} // namespace sph_native::runtime
