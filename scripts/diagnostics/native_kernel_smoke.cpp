// Compile with c++ for CPU, or hipcc -x cu -DSPH_SMOKE_HIP for NVIDIA.
// AMD: hipcc -x hip -DSPH_SMOKE_HIP --offload-arch=<target>.
#ifdef SPH_SMOKE_HIP
#include <hip/hip_runtime.h>
#endif
#include "radiusSearch/native/csrc_radius/nt_radius_forward.h"
#include "coreOperations/native/csrc_gradient/nt_gradient_forward.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifdef SPH_SMOKE_HIP
static void check(hipError_t code) {
    if (code != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(code)); std::exit(2); }
}
template <class Kernel> __global__ void run(Kernel kernel) { kernel(0); }
template <class T> T* upload(const T* host, int count) {
    T* device; check(hipMalloc(&device, sizeof(T) * count));
    check(hipMemcpy(device, host, sizeof(T) * count, hipMemcpyHostToDevice));
    return device;
}
#endif

template <class T, int D> bool qualify() {
    T positions[D * 2] = {}, support[2] = {T(1), T(1)}, mass[2] = {T(1), T(2)};
    T lo[D], hi[D], output[D] = {}, values[2] = {T(0), T(1)}, density[2] = {T(1), T(1)};
    bool periodic[D]; int32_t kinds[2] = {0, 0}, counts[1] = {2};
    int64_t offsets[1] = {0}, neighbors[2] = {0, 1}, collected[2] = {-1, -1};
    positions[D] = T(.2);
    for (int d = 0; d < D; ++d) { lo[d] = T(-1); hi[d] = T(1); periodic[d] = true; }
    sph_native::RowArgs<T, D> a{};
    a.qx = a.rx = positions; a.qh = a.rh = support; a.mass = mass;
    a.lo = lo; a.hi = hi; a.periodic = periodic; a.qkind = a.rkind = kinds;
    a.qvalue = a.rvalue = values; a.density = density;
    a.offsets = {offsets, true}; a.neighbors = {neighbors, true}; a.counts = counts;
    a.collected = collected;
    a.output = output; a.references = 2; a.support = 11; a.direction = 0; a.operation = 2;
    sph_native::CountRow<T, D>{a}(0);
    sph_native::CollectRow<T, D>{a}(0);
    if (counts[0] != 2 || collected[0] != 0 || collected[1] != 1) return false;
    sph_native::GradientRow<T, D>{a}(0);
    const T expected = output[0];
    if (!std::isfinite(expected) || expected <= T(0)) return false;
#ifdef SPH_SMOKE_HIP
    a.qx = a.rx = upload(positions, D * 2); a.qh = a.rh = upload(support, 2);
    a.mass = upload(mass, 2); a.lo = upload(lo, D); a.hi = upload(hi, D);
    a.periodic = upload(periodic, D); a.qkind = a.rkind = upload(kinds, 2);
    a.qvalue = a.rvalue = upload(values, 2); a.density = upload(density, 2);
    a.offsets = {upload(offsets, 1), true}; a.neighbors = {upload(neighbors, 2), true};
    a.counts = upload(counts, 1); a.output = upload(output, D); a.collected = upload(collected, 2);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(run<sph_native::CountRow<T, D>>), dim3(1), dim3(1), 0, 0,
                       (sph_native::CountRow<T, D>{a}));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(run<sph_native::CollectRow<T, D>>), dim3(1), dim3(1), 0, 0,
                       (sph_native::CollectRow<T, D>{a}));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(run<sph_native::GradientRow<T, D>>), dim3(1), dim3(1), 0, 0,
                       (sph_native::GradientRow<T, D>{a}));
    check(hipGetLastError()); check(hipDeviceSynchronize());
    check(hipMemcpy(output, a.output, sizeof(T) * D, hipMemcpyDeviceToHost));
    check(hipMemcpy(counts, a.counts, sizeof(counts), hipMemcpyDeviceToHost));
    check(hipMemcpy(collected, a.collected, sizeof(collected), hipMemcpyDeviceToHost));
    if (counts[0] != 2 || collected[0] != 0 || collected[1] != 1) return false;
    const void* allocations[] = {a.qx, a.qh, a.mass, a.lo, a.hi, a.periodic,
                                 a.qkind, a.qvalue, a.density, a.offsets.data,
                                 a.neighbors.data, a.counts, a.output, a.collected};
    for (const void* p : allocations) check(hipFree(const_cast<void*>(p)));
#endif
    return std::abs(output[0] - expected) <= T(1e-5) * expected;
}
int main() {
    const bool passed = qualify<float, 1>() && qualify<float, 2>() && qualify<float, 3>()
                     && qualify<double, 1>() && qualify<double, 2>() && qualify<double, 3>();
    std::puts(passed ? "Shared native kernel: float32/64, 1D/2D/3D passed" : "Shared native kernel FAILED");
    return !passed;
}
