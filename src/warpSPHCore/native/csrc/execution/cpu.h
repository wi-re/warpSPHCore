#pragma once
#include <torch/csrc/stable/ops.h>
namespace sph_native {
struct CpuLauncher {
    template <class Kernel> void operator()(int64_t n, Kernel kernel) const {
        torch::stable::parallel_for(0, n, 32, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                kernel(i);
        });
    }
};
} // namespace sph_native
