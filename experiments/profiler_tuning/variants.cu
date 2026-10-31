// Private profiler-guided candidates. Keep the readable experiment and the
// production backend as controls; no validation or allocation enters a kernel.
#include "../cluster_pairs/clusters.cu"
#include "execution/packedReference.cuh"

struct CsrCase {
    int dimension, precision, particles, index_bits;
    void *stream;
    const void *positions, *supports, *masses, *densities, *values;
    const int64_t *offsets;
    const void *indices;
    const int32_t *counts;
    const void *lower, *upper;
    const bool *periodic;
    void *packed, *output;
};

namespace profiler_tuning {
using namespace sph_native;

// Issue independent index and record loads before consuming any one record.
// Separate accumulators avoid replacing memory latency with one long FMA chain.
template <class T, int D, int Width, int Batch, bool Wide> __global__ void csr_gather(CsrCase a) {
    constexpr int pitch = ((2 * D + 3) / 4) * 4;
    const int lane = threadIdx.x % Width;
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    if (query >= a.particles)
        return;
    const auto *x = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    const QueryGeometry<T, D> geometry(x + query * D, static_cast<const T *>(a.lower),
                                       static_cast<const T *>(a.upper), a.periodic);
    const Wendland2Primal<T, D> kernel(static_cast<const T *>(a.supports)[query]);
    const int64_t begin = a.offsets[query];
    const int count = a.counts[query];
    T sums[Batch][D] = {};
    for (int base = lane; base < count; base += Width * Batch) {
        int indices[Batch];
        T records[Batch][pitch];
#pragma unroll
        for (int u = 0; u < Batch; ++u) {
            const int edge = base + u * Width;
            indices[u] = edge < count
                             ? (Wide ? int(static_cast<const int64_t *>(a.indices)[begin + edge])
                                     : static_cast<const int32_t *>(a.indices)[begin + edge])
                             : 0;
        }
#pragma unroll
        for (int u = 0; u < Batch; ++u) {
            if (base + u * Width < count) {
                if constexpr (std::is_same_v<T, float>) {
                    load_packed_reference(packed, indices[u], records[u]);
                } else {
#pragma unroll
                    for (int c = 0; c < pitch; c += 2) {
                        const auto value = reinterpret_cast<const double2 *>(
                            packed)[indices[u] * (pitch / 2) + c / 2];
                        records[u][c] = value.x;
                        records[u][c + 1] = value.y;
                    }
                }
            }
        }
#pragma unroll
        for (int u = 0; u < Batch; ++u) {
            if (base + u * Width < count) {
                T dx[D];
                const T weight = kernel.value(geometry.distance(records[u], dx));
#pragma unroll
                for (int d = 0; d < D; ++d)
                    sums[u][d] += records[u][D + d] * weight;
            }
        }
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        T sum = 0;
#pragma unroll
        for (int u = 0; u < Batch; ++u)
            sum += sums[u][d];
        const T result = group_sum<Width>(sum);
        if (lane == 0)
            static_cast<T *>(a.output)[query * D + d] = result;
    }
}

template <class T, int D> __global__ void prepare_csr(CsrCase a) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= a.particles)
        return;
    constexpr int pitch = ((2 * D + 3) / 4) * 4;
    const T volume = static_cast<const T *>(a.masses)[j] / static_cast<const T *>(a.densities)[j];
    auto *packed = static_cast<T *>(a.packed) + j * pitch;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        packed[d] = static_cast<const T *>(a.positions)[j * D + d];
        packed[D + d] = static_cast<const T *>(a.values)[j * D + d] * volume;
    }
#pragma unroll
    for (int c = 2 * D; c < pitch; ++c)
        packed[c] = 0;
}

template <class T, int D, int Width, int Batch>
void launch_csr(const CsrCase &a, int block, cudaStream_t stream) {
    const int blocks = (a.particles * Width + block - 1) / block;
    if (a.index_bits == 64)
        csr_gather<T, D, Width, Batch, true><<<blocks, block, 0, stream>>>(a);
    else
        csr_gather<T, D, Width, Batch, false><<<blocks, block, 0, stream>>>(a);
}
template <class T, int D, int Width>
void csr_batch(const CsrCase &a, int batch, int block, cudaStream_t stream) {
    switch (batch) {
    case 1:
        launch_csr<T, D, Width, 1>(a, block, stream);
        break;
    case 2:
        launch_csr<T, D, Width, 2>(a, block, stream);
        break;
    case 4:
        launch_csr<T, D, Width, 4>(a, block, stream);
        break;
    case 8:
        launch_csr<T, D, Width, 8>(a, block, stream);
        break;
    }
}
template <class T, int D>
void csr(const CsrCase &a, int batch, int width, int block, bool prepare) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (prepare)
        prepare_csr<T, D><<<(a.particles + 255) / 256, 256, 0, stream>>>(a);
    switch (width) {
    case 2:
        csr_batch<T, D, 2>(a, batch, block, stream);
        break;
    case 4:
        csr_batch<T, D, 4>(a, batch, block, stream);
        break;
    case 8:
        csr_batch<T, D, 8>(a, batch, block, stream);
        break;
    case 16:
        csr_batch<T, D, 16>(a, batch, block, stream);
        break;
    case 32:
        csr_batch<T, D, 32>(a, batch, block, stream);
        break;
    }
}

// Hardware async copy supports scalar 4/8-byte gathers even when adjacent
// clusters are not contiguous. Zero-fill partial clusters without reading them.
template <class T> __device__ void copy_async(T *shared, const T *global, bool valid) {
    const auto address = unsigned(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(address), "l"(global),
                 "n"(sizeof(T)), "r"(valid ? int(sizeof(T)) : 0)
                 : "memory");
}

template <class T, int D, int J, int Stages> struct SourceStages {
    T reference[Stages][2 * D][32 * J];
    int sizes[Stages][32], codes[Stages][32];
    uint8_t masks[Stages][32];
};

template <class T, int D, int J, int Stages, bool Async, bool PackedImages>
__device__ void stage_sources(const ClusterCase &a, int64_t base, int size, int stage,
                              SourceStages<T, D, J, Stages> &shared) {
    // Metadata and source copies are independent. Reading the small source
    // descriptor directly removes the original header-publication barrier.
    if (threadIdx.x < size) {
        const uint32_t entry = a.entries[base + threadIdx.x];
        const int source = entry >> 5;
        shared.sizes[stage][threadIdx.x] = a.source_end[source] - a.source_begin[source];
        int code = entry & 31;
        if constexpr (PackedImages) {
            // Decode base-three images once per cluster, then use cheap bit
            // fields in the pair loop without changing floating-point order.
            int bits = 0;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                bits |= (code % 3) << (2 * d);
                code /= 3;
            }
            code = bits;
        }
        shared.codes[stage][threadIdx.x] = code;
        shared.masks[stage][threadIdx.x] = a.masks[base + threadIdx.x];
    }
    const auto *packed = static_cast<const T *>(a.packed);
    for (int slot = threadIdx.x; slot < size * J; slot += blockDim.x) {
        const int source = a.entries[base + slot / J] >> 5;
        const int begin = a.source_begin[source], index = begin + slot % J;
        const bool valid = index < a.source_end[source];
#pragma unroll
        for (int d = 0; d < 2 * D; ++d) {
            const T *global = packed + d * a.particles + (valid ? index : 0);
            if constexpr (Async)
                copy_async(&shared.reference[stage][d][slot], global, valid);
            else
                shared.reference[stage][d][slot] = valid ? *global : T(0);
        }
    }
    if constexpr (Async)
        asm volatile("cp.async.commit_group;" ::: "memory");
}

template <class T, int D, int I, int J, bool Async, bool DoubleBuffer, int Unroll, bool PackedImages = false>
__global__ void staged_cluster(ClusterCase a) {
    constexpr int stages = DoubleBuffer ? 2 : 1;
    constexpr int parts = (I * J + 31) / 32;
    __shared__ SourceStages<T, D, J, stages> shared;
    const int group = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int query_bit = warp / parts, tile = a.group_first[group] + query_bit;
    const int row = (warp % parts) * (32 / J) + lane / J, column = lane % J;
    const bool valid_tile = tile < a.group_end[group];
    const int i = valid_tile ? a.query_begin[tile] + row : 0;
    const bool active = valid_tile && row < I && i < a.query_end[tile];
    T point[D], sum[D] = {};
#pragma unroll
    for (int d = 0; d < D; ++d)
        point[d] = active ? static_cast<const T *>(a.positions)[i * D + d] : T(0);
    const Wendland2Primal<T, D> kernel(active ? static_cast<const T *>(a.supports)[i] : T(1));
    const T cutoff = kernel.h * kernel.h * (T(1) + T(4) * machine_epsilon<T>);
    const int64_t end = a.offsets[group + 1];
    int64_t base = a.offsets[group];
    int stage = 0;
    if constexpr (DoubleBuffer) {
        if (base < end)
            stage_sources<T, D, J, stages, Async, PackedImages>(a, base, int(min(int64_t(32), end - base)), 0,
                                                  shared);
    }
    for (; base < end; base += 32, stage = DoubleBuffer ? 1 - stage : 0) {
        const int size = int(min(int64_t(32), end - base));
        if constexpr (DoubleBuffer) {
            if (base + 32 < end) {
                stage_sources<T, D, J, stages, Async, PackedImages>(
                    a, base + 32, int(min(int64_t(32), end - base - 32)), 1 - stage, shared);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else
                asm volatile("cp.async.wait_group 0;" ::: "memory");
        } else {
            stage_sources<T, D, J, stages, Async, PackedImages>(a, base, size, stage, shared);
            if constexpr (Async)
                asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
#pragma unroll Unroll
        for (int k = 0; k < size; ++k) {
            if (!(shared.masks[stage][k] & (1u << query_bit)))
                continue;
            int code = shared.codes[stage][k];
            T squared = 0;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const T delta =
                    (point[d] - shared.reference[stage][d][k * J + column]) - T((PackedImages ? (code & 3) : (code % 3)) - 1);
                squared += delta * delta;
                if constexpr (PackedImages) code >>= 2;
                else code /= 3;
            }
            if (active && column < shared.sizes[stage][k] && squared <= cutoff) {
                const T weight = kernel.value(scalar_sqrt(squared));
#pragma unroll
                for (int d = 0; d < D; ++d)
                    sum[d] += weight * shared.reference[stage][D + d][k * J + column];
            }
        }
        // Readers must finish before their ring slot can be reused.
        __syncthreads();
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        for (int step = J / 2; step; step /= 2)
            sum[d] += __shfl_down_sync(0xffffffffu, sum[d], step, J);
        if (active && column == 0)
            static_cast<T *>(a.output)[i * D + d] = sum[d];
    }
}

template <class T, int D, int I, int J>
void cluster_variant(const ClusterCase &a, int variant, cudaStream_t stream) {
    constexpr int threads = 8 * ((I * J + 31) / 32) * 32;
    switch (variant) {
    case 0:
        staged_cluster<T, D, I, J, false, false, 1><<<a.groups, threads, 0, stream>>>(a);
        break;
    case 1:
        staged_cluster<T, D, I, J, true, false, 1><<<a.groups, threads, 0, stream>>>(a);
        break;
    case 2:
        staged_cluster<T, D, I, J, true, true, 1><<<a.groups, threads, 0, stream>>>(a);
        break;
    case 3:
        staged_cluster<T, D, I, J, false, false, 4><<<a.groups, threads, 0, stream>>>(a);
        break;
    case 4:
        staged_cluster<T, D, I, J, false, false, 4, true><<<a.groups, threads, 0, stream>>>(a);
        break;
    case 5:
        staged_cluster<T, D, I, J, false, false, 1, true><<<a.groups, threads, 0, stream>>>(a);
        break;
    }
}
template <class T, int D> void cluster(const ClusterCase &a, int variant, bool prepare) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (prepare)
        cluster_detail::prepare_kernel<T, D><<<(a.particles + 255) / 256, 256, 0, stream>>>(a);
    if (a.query_size == 4 && a.source_size == 4)
        cluster_variant<T, D, 4, 4>(a, variant, stream);
    else if (a.query_size == 4)
        cluster_variant<T, D, 4, 8>(a, variant, stream);
    else if (a.source_size == 4)
        cluster_variant<T, D, 8, 4>(a, variant, stream);
    else
        cluster_variant<T, D, 8, 8>(a, variant, stream);
}
} // namespace profiler_tuning

extern "C" int tuned_csr(const CsrCase *a, int batch, int width, int block, int prepare,
                         int iterations) {
    for (int repeat = 0; repeat < iterations; ++repeat) {
#define RUN(T, D) profiler_tuning::csr<T, D>(*a, batch, width, block, prepare)
        if (a->precision) {
            if (a->dimension == 1)
                RUN(double, 1);
            else if (a->dimension == 2)
                RUN(double, 2);
            else
                RUN(double, 3);
        } else {
            if (a->dimension == 1)
                RUN(float, 1);
            else if (a->dimension == 2)
                RUN(float, 2);
            else
                RUN(float, 3);
        }
#undef RUN
    }
    return int(cudaGetLastError());
}
extern "C" int tuned_cluster(const ClusterCase *a, int variant, int prepare, int iterations) {
    for (int repeat = 0; repeat < iterations; ++repeat) {
#define RUN(T, D) profiler_tuning::cluster<T, D>(*a, variant, prepare)
        if (a->precision) {
            if (a->dimension == 1)
                RUN(double, 1);
            else if (a->dimension == 2)
                RUN(double, 2);
            else
                RUN(double, 3);
        } else {
            if (a->dimension == 1)
                RUN(float, 1);
            else if (a->dimension == 2)
                RUN(float, 2);
            else
                RUN(float, 3);
        }
#undef RUN
    }
    return int(cudaGetLastError());
}
