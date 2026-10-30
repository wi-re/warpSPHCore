#pragma once
#include <algorithm>
#include "torch/types.h"
#include <torch/csrc/stable/accelerator.h>
#include <cuda_runtime.h>
#include <stdexcept>
namespace sph_native {
template <class Kernel> __global__ void rows(int64_t n, Kernel kernel) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(blockDim.x) * gridDim.x) kernel(i);
}
inline cudaStream_t current_stream(const Tensor& t) {
    void* stream = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream));
    return static_cast<cudaStream_t>(stream);
}
struct GpuLauncher {
    cudaStream_t stream;
    template <class Kernel> void operator()(int64_t n, Kernel kernel) const {
        if (!n) return;
        const int blocks = int(std::min<int64_t>((n + 127) / 128, 65535));
        rows<<<blocks, 128, 0, stream>>>(n, kernel);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
};
} // namespace sph_native
