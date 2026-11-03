#pragma once
#include "api.h"
#include <vector>
#include <stdexcept>
#if defined(SPH_NATIVE_HIP)
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
using Stream = hipStream_t;
constexpr auto device_kind = sph_native::Device::HIP;
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
using Stream = cudaStream_t;
constexpr auto device_kind = sph_native::Device::CUDA;
#endif
using namespace sph_native;
static void check(GPU(Error_t) e) {
    if (e != GPU(Success))
        throw std::runtime_error(GPU(GetErrorString)(e));
}
struct Storage {
    std::vector<void *> allocations;
    Stream stream{};
    Storage() { check(GPU(StreamCreate)(&stream)); }
    ~Storage() {
        (void)GPU(StreamSynchronize)(stream);
        for (auto p : allocations)
            (void)GPU(Free)(p);
        (void)GPU(StreamDestroy)(stream);
    }
    void *allocate(size_t bytes) {
        void *p = nullptr;
        check(GPU(Malloc)(&p, bytes));
        allocations.push_back(p);
        return p;
    }
    template <class T> ConstBuffer upload(const T *p, int64_t n) {
        auto out = allocate(size_t(n) * sizeof(T));
        check(GPU(MemcpyAsync)(out, p, size_t(n) * sizeof(T), GPU(MemcpyHostToDevice), stream));
        return {out, n};
    }
    template <class T> void download(std::vector<T> &dest, Buffer src) {
        check(GPU(MemcpyAsync)(dest.data(), src.data, dest.size() * sizeof(T),
                               GPU(MemcpyDeviceToHost), stream));
        check(GPU(StreamSynchronize)(stream));
    }
};
