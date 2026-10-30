#pragma once
#if defined(SPH_NATIVE_HIP)
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
using NativeStream = hipStream_t;
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
using NativeStream = cudaStream_t;
#endif
#include <stdexcept>
namespace sph_native::runtime {
inline void check(GPU(Error_t) error) {
    if (error != GPU(Success))
        throw std::runtime_error(GPU(GetErrorString)(error));
}
struct Guard {
    int previous;
    explicit Guard(int device) {
        check(GPU(GetDevice)(&previous));
        check(GPU(SetDevice)(device));
    }
    ~Guard() { (void)GPU(SetDevice)(previous); }
};
} // namespace sph_native::runtime
