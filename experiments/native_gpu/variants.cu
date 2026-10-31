// Private CUDA timing ABI for controlled scheduling/arithmetic experiments.
// No allocations, synchronization, Torch calls or content validation happen here.
#include "execution/valueTraversal.cuh"
#include <cuda_runtime.h>
#include <cstdint>
using namespace sph_native;
// ABI 2 descriptor from scripts/benchmarks/bench_backends.py (NativeCase).
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
namespace {
template <class T> RowArgs<T, 2> arguments(const Case &c) {
    RowArgs<T, 2> a{};
    a.qx = a.rx = static_cast<const T *>(c.positions);
    a.qh = a.rh = static_cast<const T *>(c.supports);
    a.mass = static_cast<const T *>(c.masses);
    a.density = static_cast<const T *>(c.densities);
    a.qvalue = a.rvalue = static_cast<const T *>(c.values);
    a.lo = static_cast<const T *>(c.lower);
    a.hi = static_cast<const T *>(c.upper);
    a.periodic = c.periodic;
    a.qkind = a.rkind = c.kinds;
    a.counts = const_cast<int32_t *>(c.counts);
    a.offsets = {c.offsets, true};
    a.neighbors = {c.indices, true};
    a.output = static_cast<T *>(c.output);
    a.references = c.particles;
    a.fields = a.outputs = 2;
    a.kernel = a.direction = 0;
    a.support = 11;
    return a;
}
template <class T, int Kind> __global__ void prepare(int64_t n, RowArgs<T, 2> a, T *out) {
    for (int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; j < n;
         j += int64_t(blockDim.x) * gridDim.x) {
        const T volume = a.mass[j] / a.density[j];
        if constexpr (Kind == 2) {
            out[4*j] = a.rx[2*j];
            out[4*j+1] = a.rx[2*j+1];
            out[4*j+2] = a.rvalue[2*j] * volume;
            out[4*j+3] = a.rvalue[2*j+1] * volume;
        } else if constexpr (Kind == 1) {
            out[2*j] = a.rvalue[2*j] * volume;
            out[2*j+1] = a.rvalue[2*j+1] * volume;
        } else {
            out[j] = volume;
        }
    }
}
template <class T> struct Vec4;
template <> struct Vec4<float> { using type = float4; };
template <> struct Vec4<double> { using type = double4; };
template <class T> struct Vec;
template <> struct Vec<float> { using type = float2; };
template <> struct Vec<double> { using type = double2; };

template <class T, int Width, int Mode> struct LeanRow {
    RowArgs<T, 2> a;
    const T *temporary;
    __device__ void operator()(int64_t i, int lane) const {
        constexpr int unroll = Mode == 5 ? 2 : (Mode == 6 ? 4 : 1);
        T sums[unroll][2] = {};
        const QueryGeometry<T, 2> geometry(a.qx + 2*i, a.lo, a.hi, a.periodic);
        const Wendland2Primal<T, 2> kernel(a.qh[i]);
        const int64_t begin = static_cast<const int64_t *>(a.offsets.data)[i];
        const int count = a.counts[i];
        for (int base = lane; base < count; base += Width * unroll) {
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                const int edge = base + u * Width;
                if (edge >= count)
                    continue;
                const int64_t raw = static_cast<const int64_t *>(a.neighbors.data)[begin + edge];
                using J = typename std::conditional<Mode == 1, int64_t, int>::type;
                const J j = static_cast<J>(raw);
                T dx[2], r, first, second;
                if constexpr (Mode == 11) {
                    using V = typename Vec4<T>::type;
                    const V point = reinterpret_cast<const V *>(temporary)[j];
                    const T xy[2] = {point.x, point.y};
                    r = geometry.distance(xy, dx);
                    first = point.z;
                    second = point.w;
                } else if constexpr ((Mode == 8 || Mode == 9 || Mode == 10)) {
                    using V = typename Vec<T>::type;
                    const V position = reinterpret_cast<const V *>(a.rx)[j];
                    const T xy[2] = {position.x, position.y};
                    r = geometry.distance(xy, dx);
                    const V field = reinterpret_cast<const V *>(Mode == 9 ? temporary : a.rvalue)[j];
                    first = field.x;
                    second = field.y;
                } else {
                    r = geometry.distance(a.rx + 2*j, dx);
                    first = a.rvalue[2*j];
                    second = a.rvalue[2*j+1];
                }
                const T weight = kernel.value(r);
                if constexpr (Mode == 4) {
                    first = temporary[2*j];
                    second = temporary[2*j+1];
                } else if constexpr (Mode != 9 && Mode != 11) {
                    const T volume = (Mode == 3 || Mode == 10) ? temporary[j] : a.mass[j] / a.density[j];
                    first *= volume;
                    second *= volume;
                }
                sums[u][0] += first * weight;
                sums[u][1] += second * weight;
            }
        }
        T first = 0, second = 0;
#pragma unroll
        for (int u = 0; u < unroll; ++u) {
            first += sums[u][0];
            second += sums[u][1];
        }
        first = group_sum<Width>(first);
        second = group_sum<Width>(second);
        if (lane == 0) {
            a.output[2*i] = first;
            a.output[2*i+1] = second;
        }
    }
};
template <int Width, int Block, class Row> void launch(int64_t n, Row row, cudaStream_t stream) {
    const int blocks = int(std::min<int64_t>((n*Width + Block-1)/Block, 65535));
    cooperative_rows<Width><<<blocks, Block, 0, stream>>>(n, row);
}
template <class T, int Width, int Block> void mode(const Case &c, int value, T *temporary) {
    const auto a = arguments<T>(c);
    const auto stream = static_cast<cudaStream_t>(c.stream);
    if (value == 0) {
        launch<Width, Block>(c.particles, CooperativeValues<T,2,2,Width,false,true>{a}, stream);
        return;
    }
    if (value == 3 || value == 10)
        prepare<T,0><<<std::min<int64_t>((c.particles+255)/256,65535),256,0,stream>>>(c.particles,a,temporary);
    if (value == 4 || value == 9)
        prepare<T,1><<<std::min<int64_t>((c.particles+255)/256,65535),256,0,stream>>>(c.particles,a,temporary);
    if (value == 11)
        prepare<T,2><<<std::min<int64_t>((c.particles+255)/256,65535),256,0,stream>>>(c.particles,a,temporary);
#define EXPERIMENT(M) case M: launch<Width, Block>(c.particles, LeanRow<T,Width,M>{a,temporary},stream); break
    switch (value) {
        EXPERIMENT(1); EXPERIMENT(2); EXPERIMENT(3); EXPERIMENT(4);
        EXPERIMENT(5); EXPERIMENT(6); EXPERIMENT(8); EXPERIMENT(9); EXPERIMENT(10); EXPERIMENT(11);
    }
#undef EXPERIMENT
}
template <class T, int Width> void block(const Case &c, int value, int size, T *temporary) {
    if (size == 64) mode<T,Width,64>(c,value,temporary);
    else if (size == 256) mode<T,Width,256>(c,value,temporary);
    else mode<T,Width,128>(c,value,temporary);
}
template <class T> void width(const Case &c, int value, int lanes, int size, T *temporary) {
#define LANES(W) case W: block<T,W>(c,value,size,temporary); break
    switch (lanes) { LANES(1); LANES(2); LANES(4); LANES(8); LANES(16); LANES(32); }
#undef LANES
}
}
extern "C" __attribute__((visibility("default"))) int value_variant(const Case *c, int mode, int lanes, int block, void *temporary) {
    if (c->dimension != 2 || c->particles > INT32_MAX) return int(cudaErrorInvalidValue);
    if (c->precision == 0) width<float>(*c,mode,lanes,block,static_cast<float *>(temporary));
    else width<double>(*c,mode,lanes,block,static_cast<double *>(temporary));
    return int(cudaGetLastError());
}
extern "C" __attribute__((visibility("default"))) int value_variant_batch(const Case *c, int mode, int lanes, int block,
                                   void *temporary, int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int status = value_variant(c, mode, lanes, block, temporary);
        if (status) return status;
    }
    return 0;
}
