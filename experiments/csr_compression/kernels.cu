// Lossless particle-CSR encodings, isolated from production dispatch and AD.
// A subgroup owns a query row; independent chunks bound decode dependencies.
#include "execution/packedReference.cuh"
#include "kernels/kernelFunctions/wendland2.h"
#include "math/queryGeometry.h"
#include <cuda_runtime.h>
#include <cstdint>

struct CompressionCase {
    int dimension, precision, queries, references;
    void *stream;
    const void *positions, *supports, *masses, *densities, *values;
    const void *lower, *upper;
    const bool *periodic;
    const int64_t *row_offsets, *indices;
    const int32_t *counts;
    const int64_t *row_blocks;
    const uint32_t *particle_bases;
    uint8_t *codes;
    uint32_t *bases, *masks, *meta_counts, *meta_offsets, *extras;
    void *packed, *output;
    int64_t *decoded;
};

namespace compressed_csr {
using namespace sph_native;
enum Format { Raw64, Raw32, Delta, Cell };

template <int Width> __device__ unsigned subgroup_mask() {
    constexpr unsigned bits = 0xffffffffu >> (32 - Width);
    return bits << ((threadIdx.x & 31) & ~(Width - 1));
}
template <int Width> __device__ uint32_t inclusive_sum(uint32_t value) {
    const int lane = threadIdx.x & (Width - 1);
#pragma unroll
    for (int shift = 1; shift < Width; shift *= 2) {
        const uint32_t previous = __shfl_up_sync(subgroup_mask<Width>(), value, shift, Width);
        if (lane >= shift)
            value += previous;
    }
    return value;
}
__device__ unsigned lower_lanes(int lane) { return (1u << lane) - 1; }
__device__ unsigned inclusive_lanes(int lane) { return 0xffffffffu >> (31 - lane); }

// A source cell may be arbitrarily populated. Partition its contiguous particle
// range into <=256-entry pieces, preserving a byte local index without overflow.
__global__ void make_particle_bases(int cells, const int64_t *offsets, uint32_t *bases) {
    const int cell = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (cell >= cells)
        return;
    const int64_t begin = offsets[cell], end = offsets[cell + 1];
    for (int64_t j = begin + lane; j < end; j += 32)
        bases[j] = uint32_t(begin + ((j - begin) / 256) * 256);
}

// Count variable payloads, prefix-sum them outside the kernel, then fill. Only
// planning touches the input CSR or the transient particle-to-cell base map.
template <int Width, int Encoding, bool Fill> __global__ void encode(CompressionCase a) {
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (query >= a.queries)
        return;
    const int count = a.counts[query];
    for (int start = 0; start < count; start += Width) {
        const int64_t block = a.row_blocks[query] + start / Width;
        const bool valid = start + lane < count;
        const uint32_t j = valid ? uint32_t(a.indices[a.row_offsets[query] + start + lane]) : 0;
        if constexpr (Encoding == Delta) {
            const uint32_t previous = __shfl_up_sync(subgroup_mask<Width>(), j, 1, Width);
            const uint32_t delta = valid && lane ? j - previous : 0;
            const unsigned escapes = group_ballot<Width>(delta >= 255);
            if constexpr (Fill) {
                if (!lane)
                    a.bases[block] = j;
                a.codes[block * Width + lane] = uint8_t(delta >= 255 ? 255 : delta);
                if (delta >= 255)
                    a.extras[a.meta_offsets[block] + __popc(escapes & lower_lanes(lane))] = delta;
            } else if (!lane) {
                a.meta_counts[block] = __popc(escapes);
            }
        } else {
            const uint32_t base = valid ? a.particle_bases[j] : 0;
            const uint32_t previous = __shfl_up_sync(subgroup_mask<Width>(), base, 1, Width);
            const bool changed = valid && (!lane || base != previous);
            const unsigned starts = group_ballot<Width>(changed);
            if constexpr (Fill) {
                if (!lane)
                    a.masks[block] = starts;
                a.codes[block * Width + lane] = valid ? uint8_t(j - base) : 0;
                if (changed)
                    a.extras[a.meta_offsets[block] + __popc(starts & lower_lanes(lane))] = base;
            } else if (!lane) {
                a.meta_counts[block] = __popc(starts);
            }
        }
    }
}

template <int Width, int Encoding>
__device__ int decode(const CompressionCase &a, int64_t block, int lane) {
    const uint32_t code = a.codes[block * Width + lane];
    if constexpr (Encoding == Delta) {
        const unsigned escapes = group_ballot<Width>(code == 255);
        uint32_t delta = code;
        if (code == 255)
            delta = a.extras[a.meta_offsets[block] + __popc(escapes & lower_lanes(lane))];
        return int(a.bases[block] + inclusive_sum<Width>(delta));
    } else {
        const unsigned starts = a.masks[block];
        const int run = __popc(starts & inclusive_lanes(lane)) - 1;
        return int(a.extras[a.meta_offsets[block] + run] + code);
    }
}

// Four byte codes arrive in one aligned uint32 load. Delta scans operate on
// four-entry totals across fewer lanes; cell runs reuse a base within a quartet.
template <int Chunk, int Encoding>
__device__ void decode_four(const CompressionCase &a, int64_t block, int lane, int (&ids)[4]) {
    constexpr int Width = Chunk / 4;
    const uint32_t word = reinterpret_cast<const uint32_t *>(a.codes)[block * Width + lane];
    uint32_t codes[4];
#pragma unroll
    for (int u = 0; u < 4; ++u)
        codes[u] = (word >> (8 * u)) & 255;
    if constexpr (Encoding == Delta) {
        unsigned escapes = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u)
            escapes += codes[u] == 255;
        const unsigned end = inclusive_sum<Width>(escapes);
        unsigned cursor = a.meta_offsets[block] + end - escapes;
        uint32_t local[4], total = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint32_t delta = codes[u] == 255 ? a.extras[cursor++] : codes[u];
            total += delta;
            local[u] = total;
        }
        const uint32_t before = inclusive_sum<Width>(total) - total;
        const uint32_t base = a.bases[block] + before;
#pragma unroll
        for (int u = 0; u < 4; ++u)
            ids[u] = int(base + local[u]);
    } else {
        const unsigned starts = a.masks[block];
        const uint32_t offset = a.meta_offsets[block];
        uint32_t base = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int position = lane * 4 + u;
            if (u == 0 || (starts & (1u << position))) {
                const int run = __popc(starts & inclusive_lanes(position)) - 1;
                base = a.extras[offset + run];
            }
            ids[u] = int(base + codes[u]);
        }
    }
}

template <int Chunk, int Encoding> __global__ void decode_rows_four(CompressionCase a) {
    constexpr int Width = Chunk / 4;
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (query >= a.queries)
        return;
    for (int start = 0; start < a.counts[query]; start += Chunk) {
        int ids[4];
        decode_four<Chunk, Encoding>(a, a.row_blocks[query] + start / Chunk, lane, ids);
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (start + lane * 4 + u < a.counts[query])
                a.decoded[a.row_offsets[query] + start + lane * 4 + u] = ids[u];
    }
}

template <int Width, int Encoding> __global__ void decode_rows(CompressionCase a) {
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (query >= a.queries)
        return;
    for (int start = 0; start < a.counts[query]; start += Width) {
        const int j = decode<Width, Encoding>(a, a.row_blocks[query] + start / Width, lane);
        if (start + lane < a.counts[query])
            a.decoded[a.row_offsets[query] + start + lane] = j;
    }
}

template <class T, int D> __global__ void prepare(CompressionCase a) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= a.references)
        return;
    constexpr int pitch = ((2 * D + 3) / 4) * 4;
    const T volume = static_cast<const T *>(a.masses)[j] / static_cast<const T *>(a.densities)[j];
    T *packed = static_cast<T *>(a.packed) + int64_t(j) * pitch;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        packed[d] = static_cast<const T *>(a.positions)[int64_t(j) * D + d];
        packed[D + d] = static_cast<const T *>(a.values)[int64_t(j) * D + d] * volume;
    }
#pragma unroll
    for (int d = 2 * D; d < pitch; ++d)
        packed[d] = 0;
}

// Raw32/64 controls share the same math, packing, scheduling and accumulators.
// Batched chunks expose independent gathers before consuming their records.
template <class T, int D, int Width, int Batch, int Encoding>
__global__ void interpolate(CompressionCase a) {
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (query >= a.queries)
        return;
    constexpr int pitch = ((2 * D + 3) / 4) * 4;
    const auto *x = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    const QueryGeometry<T, D> geometry(x + int64_t(query) * D, static_cast<const T *>(a.lower),
                                     static_cast<const T *>(a.upper), a.periodic);
    const Wendland2Primal<T, D> kernel(static_cast<const T *>(a.supports)[query]);
    const int count = a.counts[query];
    T sums[Batch][D] = {};
    for (int start = 0; start < count; start += Width * Batch) {
        int ids[Batch];
        T records[Batch][pitch];
#pragma unroll
        for (int u = 0; u < Batch; ++u) {
            const int edge = start + u * Width + lane;
            if constexpr (Encoding < Delta) {
                if (edge < count) {
                    const int64_t offset = a.row_offsets[query] + edge;
                    ids[u] = Encoding == Raw64 ? int(a.indices[offset])
                                              : reinterpret_cast<const int32_t *>(a.indices)[offset];
                }
            } else {
                // All subgroup lanes participate in decode, including padding;
                // a whole absent chunk is skipped uniformly by the subgroup.
                if (start + u * Width < count)
                    ids[u] = decode<Width, Encoding>(a, a.row_blocks[query] + start / Width + u, lane);
            }
        }
#pragma unroll
        for (int u = 0; u < Batch; ++u)
            if (start + u * Width + lane < count)
                load_packed_reference(packed, ids[u], records[u]);
#pragma unroll
        for (int u = 0; u < Batch; ++u) {
            if (start + u * Width + lane < count) {
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
        const T total = group_sum<Width>(sum);
        if (!lane)
            static_cast<T *>(a.output)[int64_t(query) * D + d] = total;
    }
}

template <int Width, int Encoding>
void codec(const CompressionCase &a, int action, cudaStream_t stream) {
    const int lanes = action == 3 ? Width / 4 : Width;
    const int blocks = (int64_t(a.queries) * lanes + 127) / 128;
    if (!blocks)
        return;
    if (action == 0)
        encode<Width, Encoding, false><<<blocks, 128, 0, stream>>>(a);
    else if (action == 1)
        encode<Width, Encoding, true><<<blocks, 128, 0, stream>>>(a);
    else if (action == 2)
        decode_rows<Width, Encoding><<<blocks, 128, 0, stream>>>(a);
    else
        decode_rows_four<Width, Encoding><<<blocks, 128, 0, stream>>>(a);
}
template <int Width>
void codec_format(const CompressionCase &a, int encoding, int action, cudaStream_t stream) {
    if (encoding == Delta)
        codec<Width, Delta>(a, action, stream);
    else
        codec<Width, Cell>(a, action, stream);
}
template <class T, int D, int Width, int Batch>
void launch(const CompressionCase &a, int format, int threads, cudaStream_t stream) {
    const int blocks = (int64_t(a.queries) * Width + threads - 1) / threads;
#define RUN(F) interpolate<T, D, Width, Batch, F><<<blocks, threads, 0, stream>>>(a)
    switch (format) {
    case Raw64: RUN(Raw64); break;
    case Raw32: RUN(Raw32); break;
    case Delta: RUN(Delta); break;
    case Cell: RUN(Cell); break;
    }
#undef RUN
}
template <class T, int D, int Width>
void batch(const CompressionCase &a, int format, int batching, int threads, cudaStream_t stream) {
    if (batching == 1)
        launch<T, D, Width, 1>(a, format, threads, stream);
    else if (batching == 2)
        launch<T, D, Width, 2>(a, format, threads, stream);
    else
        launch<T, D, Width, 4>(a, format, threads, stream);
}
template <class T, int D>
void run(const CompressionCase &a, int format, int width, int batching, int threads, bool pack) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (pack && a.references)
        prepare<T, D><<<(a.references + 255) / 256, 256, 0, stream>>>(a);
    if (!a.queries)
        return;
    switch (width) {
    case 4: batch<T, D, 4>(a, format, batching, threads, stream); break;
    case 8: batch<T, D, 8>(a, format, batching, threads, stream); break;
    case 16: batch<T, D, 16>(a, format, batching, threads, stream); break;
    case 32: batch<T, D, 32>(a, format, batching, threads, stream); break;
    }
}
template <class T, int D, int Chunk, int Batch, int Encoding>
__global__ void interpolate_four(CompressionCase a) {
    constexpr int Width = Chunk / 4, Slots = Batch * 4;
    constexpr int pitch = ((2 * D + 3) / 4) * 4;
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (query >= a.queries)
        return;
    const auto *x = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    const QueryGeometry<T, D> geometry(x + int64_t(query) * D, static_cast<const T *>(a.lower),
                                     static_cast<const T *>(a.upper), a.periodic);
    const Wendland2Primal<T, D> kernel(static_cast<const T *>(a.supports)[query]);
    const int count = a.counts[query];
    T sums[Slots][D] = {};
    for (int start = 0; start < count; start += Chunk * Batch) {
        int ids[Batch][4];
        T records[Slots][pitch];
#pragma unroll
        for (int b = 0; b < Batch; ++b) {
            if constexpr (Encoding < Delta) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int edge = start + b * Chunk + lane * 4 + u;
                    if (edge < count) {
                        const int64_t offset = a.row_offsets[query] + edge;
                        ids[b][u] = Encoding == Raw64 ? int(a.indices[offset])
                                           : reinterpret_cast<const int32_t *>(a.indices)[offset];
                    }
                }
            } else if (start + b * Chunk < count) {
                decode_four<Chunk, Encoding>(a, a.row_blocks[query] + start / Chunk + b, lane, ids[b]);
            }
        }
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int edge = start + (slot / 4) * Chunk + lane * 4 + slot % 4;
            if (edge < count)
                load_packed_reference(packed, ids[slot / 4][slot % 4], records[slot]);
        }
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int edge = start + (slot / 4) * Chunk + lane * 4 + slot % 4;
            if (edge < count) {
                T dx[D];
                const T weight = kernel.value(geometry.distance(records[slot], dx));
#pragma unroll
                for (int d = 0; d < D; ++d)
                    sums[slot][d] += records[slot][D + d] * weight;
            }
        }
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        T sum = 0;
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot)
            sum += sums[slot][d];
        const T total = group_sum<Width>(sum);
        if (!lane)
            static_cast<T *>(a.output)[int64_t(query) * D + d] = total;
    }
}

template <class T, int D, int Chunk, int Batch>
void launch_four(const CompressionCase &a, int format, int threads, cudaStream_t stream) {
    const int blocks = (int64_t(a.queries) * (Chunk / 4) + threads - 1) / threads;
#define RUN(F) interpolate_four<T, D, Chunk, Batch, F><<<blocks, threads, 0, stream>>>(a)
    switch (format) {
    case Raw64: RUN(Raw64); break;
    case Raw32: RUN(Raw32); break;
    case Delta: RUN(Delta); break;
    case Cell: RUN(Cell); break;
    }
#undef RUN
}
template <class T, int D, int Chunk>
void batch_four(const CompressionCase &a, int format, int batching, int threads, cudaStream_t stream) {
    if (batching == 1)
        launch_four<T, D, Chunk, 1>(a, format, threads, stream);
    else
        launch_four<T, D, Chunk, 2>(a, format, threads, stream);
}
template <class T, int D>
void run_four(const CompressionCase &a, int format, int chunk, int batching, int threads, bool pack) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (pack && a.references)
        prepare<T, D><<<(a.references + 255) / 256, 256, 0, stream>>>(a);
    if (!a.queries)
        return;
    switch (chunk) {
    case 4: batch_four<T, D, 4>(a, format, batching, threads, stream); break;
    case 8: batch_four<T, D, 8>(a, format, batching, threads, stream); break;
    case 16: batch_four<T, D, 16>(a, format, batching, threads, stream); break;
    case 32: batch_four<T, D, 32>(a, format, batching, threads, stream); break;
    }
}

} // namespace compressed_csr

extern "C" int compression_cell_bases(int cells, const int64_t *offsets, uint32_t *bases, void *stream) {
    if (cells)
        compressed_csr::make_particle_bases<<<(int64_t(cells) * 32 + 127) / 128, 128, 0,
                                            static_cast<cudaStream_t>(stream)>>>(cells, offsets, bases);
    return int(cudaGetLastError());
}
extern "C" int compression_codec(const CompressionCase *a, int format, int width, int action) {
    const auto stream = static_cast<cudaStream_t>(a->stream);
    switch (width) {
    case 4: compressed_csr::codec_format<4>(*a, format, action, stream); break;
    case 8: compressed_csr::codec_format<8>(*a, format, action, stream); break;
    case 16: compressed_csr::codec_format<16>(*a, format, action, stream); break;
    case 32: compressed_csr::codec_format<32>(*a, format, action, stream); break;
    }
    return int(cudaGetLastError());
}
extern "C" int compression_interpolate(const CompressionCase *a, int format, int width, int batch,
                                       int threads, int pack, int iterations) {
    for (int repeat = 0; repeat < iterations; ++repeat) {
#define RUN(T, D) compressed_csr::run<T, D>(*a, format, width, batch, threads, pack)
        if (a->precision) {
            if (a->dimension == 1) { RUN(double, 1); }
            else if (a->dimension == 2) { RUN(double, 2); }
            else { RUN(double, 3); }
        } else {
            if (a->dimension == 1) { RUN(float, 1); }
            else if (a->dimension == 2) { RUN(float, 2); }
            else { RUN(float, 3); }
        }
#undef RUN
    }
    return int(cudaGetLastError());
}

extern "C" int compression_interpolate_vec4(const CompressionCase *a, int format, int width, int batch,
                                       int threads, int pack, int iterations) {
    for (int repeat = 0; repeat < iterations; ++repeat) {
#define RUN(T, D) compressed_csr::run_four<T, D>(*a, format, width, batch, threads, pack)
        if (a->precision) {
            if (a->dimension == 1) { RUN(double, 1); }
            else if (a->dimension == 2) { RUN(double, 2); }
            else { RUN(double, 3); }
        } else {
            if (a->dimension == 1) { RUN(float, 1); }
            else if (a->dimension == 2) { RUN(float, 2); }
            else { RUN(float, 3); }
        }
#undef RUN
    }
    return int(cudaGetLastError());
}
