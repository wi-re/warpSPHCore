// A scalar CUDA-core FMA ceiling, independent of Torch and tensor cores.
#include <cuda_profiler_api.h>
// Eight independent recurrence chains expose instruction-level parallelism.
// This is a best-case arithmetic workload, not an SPH performance prediction.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

template <class T> __global__ void scalar_fma_ceiling(T* output, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    T accumulator[8];
    #pragma unroll
    for (int chain = 0; chain < 8; ++chain)
        accumulator[chain] = T(0.01 * (chain + 1)) + T(index & 255) * T(0.0001);
    #pragma unroll 32
    for (int iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll
        for (int chain = 0; chain < 8; ++chain) {
            // Explicit instructions prevent contraction/elimination ambiguity.
            if constexpr (std::is_same_v<T, float>)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(accumulator[chain]) : "f"(T(0.9999)), "f"(T(0.00001)));
            else
                asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(accumulator[chain]) : "d"(T(0.9999)), "d"(T(0.00001)));
        }
    }
    T sum = T(0);
    #pragma unroll
    for (int chain = 0; chain < 8; ++chain) sum += accumulator[chain];
    output[index] = sum;
}

template <class T> void measure(int blocks, bool profile) {
    constexpr int threads = 256, iterations = 4096;
    T* output = nullptr;
    check(cudaMalloc(&output, size_t(blocks) * threads * sizeof(T)));
    for (int warmup = 0; warmup < 5; ++warmup)
        scalar_fma_ceiling<<<blocks, threads>>>(output, iterations);
    check(cudaDeviceSynchronize());
    std::vector<float> samples;
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    for (int repeat = 0; repeat < 9; ++repeat) {
        check(cudaEventRecord(begin));
        for (int batch = 0; batch < 5; ++batch)
            scalar_fma_ceiling<<<blocks, threads>>>(output, iterations);
        check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, begin, end));
        samples.push_back(elapsed / 5);
    }
    if (profile) {
        check(cudaProfilerStart());
        scalar_fma_ceiling<<<blocks, threads>>>(output, iterations);
        check(cudaDeviceSynchronize());
        check(cudaProfilerStop());
    }
    std::sort(samples.begin(), samples.end());
    const double flops = double(blocks) * threads * iterations * 8 * 2;
    std::printf("{\"precision\":\"%s\",\"fma_flops\":%.0f,\"event_ms\":%.8f,\"tflops\":%.8f,\"samples_ms\":[", sizeof(T)==4 ? "float32":"float64", flops, samples[4], flops/samples[4]/1e9);
    for (size_t i=0; i<samples.size(); ++i) std::printf("%s%.8f", i ? ",":"", samples[i]);
    std::printf("]}\n");
    check(cudaFree(output));
    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));
}

int main(int argc, char** argv) {
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    const bool profile = argc > 1 && std::strcmp(argv[1], "--profile") == 0;
    measure<float>(device.multiProcessorCount*8, profile);
    measure<double>(device.multiProcessorCount*8, profile);
}
