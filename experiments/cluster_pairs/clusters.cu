// Experimental directed cluster lists. Reuse the cell planner and the exact
// production Wendland helper; no production dispatch or AD source is changed.
#include "../cell_tiles/tiles.cu"

struct ClusterCase {
    int dimension, precision, particles, queries, sources, groups;
    int query_size, source_size, group_size;
    void *stream;
    const void *positions, *supports, *masses, *densities, *values;
    const int32_t *query_begin, *query_end, *query_cell;
    const int32_t *source_begin, *source_end;
    const int64_t *source_cell_offsets;
    const int32_t *group_first, *group_end, *group_cell;
    const int32_t *query_group, *query_bit, *neighbors, *images;
    void *query_bounds, *source_bounds;
    int64_t *counts, *offsets;
    uint32_t *entries;
    uint8_t *masks;
    int64_t *byte_counts, *byte_offsets;
    uint8_t *compressed, *image_codes;
    void *packed, *output;
};

namespace cluster_detail {
using namespace sph_native;
constexpr unsigned full_warp = 0xffffffffu;

// One conservative bounding box and maximum Gather radius per cluster. Clusters
// never cross a physical cell boundary, so periodic images have no ambiguity.
template <class T, int D> __global__ void bounds_kernel(ClusterCase a, bool source) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (source ? a.sources : a.queries))
        return;
    const auto *x = static_cast<const T *>(a.positions);
    const auto *h = static_cast<const T *>(a.supports);
    const int begin = source ? a.source_begin[c] : a.query_begin[c];
    const int end = source ? a.source_end[c] : a.query_end[c];
    auto *box = static_cast<T *>(source ? a.source_bounds : a.query_bounds) + c * (2 * D + 1);
    T maximum_h = 0;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        T lo = x[begin * D + d], hi = lo;
        for (int i = begin; i < end; ++i) {
            lo = min(lo, x[i * D + d]);
            hi = max(hi, x[i * D + d]);
        }
        box[d] = lo;
        box[D + d] = hi;
    }
    for (int i = begin; i < end; ++i)
        maximum_h = max(maximum_h, h[i]);
    box[2 * D] = maximum_h;
}

template <class T, int D>
__device__ uint8_t interaction_mask(const ClusterCase &a, int group, int source,
                                    const int (&image)[D], bool exact) {
    const auto *x = static_cast<const T *>(a.positions);
    const auto *h = static_cast<const T *>(a.supports);
    const auto *source_box = static_cast<const T *>(a.source_bounds) + source * (2 * D + 1);
    uint8_t mask = 0;
    const T outward = T(1) + T(16) * machine_epsilon<T>;
    for (int q = a.group_first[group]; q < a.group_end[group]; ++q) {
        const auto *box = static_cast<const T *>(a.query_bounds) + q * (2 * D + 1);
        T lower_squared = 0;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            // Preserve pair-kernel subtraction before applying the image.
            // Adding an image first can lose an extra FP32 bit near 1.
            const T gap = max(T(0), max(box[d] - source_box[D + d] - T(image[d]),
                                        (source_box[d] - box[D + d]) + T(image[d])));
            lower_squared += gap * gap;
        }
        if (lower_squared > box[2 * D] * box[2 * D] * outward)
            continue;
        bool keep = !exact;
        // Exact cluster pruning costs more to build, but eliminates cluster
        // pairs whose bounding boxes overlap without any valid particle pair.
        for (int i = a.query_begin[q]; exact && !keep && i < a.query_end[q]; ++i) {
            for (int j = a.source_begin[source]; j < a.source_end[source]; ++j) {
                T squared = 0;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    const T dx = (x[i * D + d] - x[j * D + d]) - T(image[d]);
                    squared += dx * dx;
                }
                if (squared <= h[i] * h[i] * outward) {
                    keep = true;
                    break;
                }
            }
        }
        if (keep)
            mask |= uint8_t(1u << (q - a.group_first[group]));
    }
    return mask;
}

// A fixed 128-thread inclusive scan. The same deterministic traversal is used
// by count and fill; sorting occupied neighbor cells gives ascending source IDs.
__device__ int block_scan(int value, int (&warp_totals)[4]) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int step = 1; step < 32; step *= 2) {
        const int previous = __shfl_up_sync(full_warp, value, step);
        if (lane >= step)
            value += previous;
    }
    if (lane == 31)
        warp_totals[warp] = value;
    __syncthreads();
    for (int w = 0; w < warp; ++w)
        value += warp_totals[w];
    // Readers must finish before the next scan overwrites these four slots.
    __syncthreads();
    return value;
}

template <class T, int D, bool Fill> __global__ void build_kernel(ClusterCase a, bool exact) {
    __shared__ int warp_totals[4];
    const int group = blockIdx.x, cell = a.group_cell[group];
    int cursor = 0;
    for (int s = 0; s < stencil_size<D>; ++s) {
        const int neighbor = a.neighbors[cell * stencil_size<D> + s];
        if (neighbor < 0)
            continue;
        int image[D], code = 0, power = 1;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            image[d] = a.images[(cell * stencil_size<D> + s) * D + d];
            code += (image[d] + 1) * power;
            power *= 3;
        }
        const int begin = int(a.source_cell_offsets[neighbor]);
        const int end = int(a.source_cell_offsets[neighbor + 1]);
        for (int base = begin; base < end; base += 128) {
            const int source = base + threadIdx.x;
            const uint8_t mask =
                source < end ? interaction_mask<T, D>(a, group, source, image, exact) : 0;
            const int prefix = block_scan(mask != 0, warp_totals);
            const int total = __shfl_sync(full_warp, prefix, 31);
            // The last warp holds the full block total; broadcast through SM.
            if (threadIdx.x == 127)
                warp_totals[0] = total;
            __syncthreads();
            const int accepted = warp_totals[0];
            if constexpr (Fill) {
                if (mask) {
                    const auto out = a.offsets[group] + cursor + prefix - 1;
                    a.entries[out] = (uint32_t(source) << 5) | uint32_t(code);
                    a.masks[out] = mask;
                }
            }
            cursor += accepted;
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        a.counts[group] = cursor;
}

__device__ int nibble_count(uint32_t value) { return max(1, (32 - __clz(value) + 3) / 4); }
__device__ uint8_t read_nibble(const uint8_t *data, int index) {
    return (data[index / 2] >> (4 * (index % 2))) & 15;
}
__device__ void write_nibble(uint8_t *data, int index, uint32_t value) {
    if (index % 2 == 0)
        data[index / 2] = uint8_t(value & 15);
    else
        data[index / 2] |= uint8_t((value & 15) << 4);
}

// Paper-inspired 32-entry delta blocks: a unity bitmask, one info nibble per
// other delta, then variable-length payload nibbles. Images and query masks
// remain separate byte arrays. This avoids hiding their cost in memory reports.
template <bool Fill> __global__ void compress_kernel(ClusterCase a) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= a.groups)
        return;
    int bytes = 0;
    uint32_t previous = 0;
    for (int64_t base = a.offsets[group]; base < a.offsets[group + 1]; base += 32) {
        const int size = int(min(int64_t(32), a.offsets[group + 1] - base));
        uint32_t unity = 0, delta[32];
        int infos = 0, payload = 0;
        for (int k = 0; k < size; ++k) {
            const uint32_t entry = a.entries[base + k], id = entry >> 5;
            // +1 lets the first cluster ID (including zero) use a positive delta.
            delta[k] = id + 1 - previous;
            previous = id + 1;
            if (delta[k] == 1)
                unity |= 1u << k;
            else {
                ++infos;
                if (delta[k] > 9)
                    payload += nibble_count(delta[k]);
            }
            if constexpr (Fill)
                a.image_codes[base + k] = entry & 31;
        }
        if constexpr (Fill) {
            auto *out = a.compressed + a.byte_offsets[group] + bytes;
            for (int b = 0; b < 4; ++b)
                out[b] = uint8_t(unity >> (8 * b));
            auto *info = out + 4;
            auto *data = info + (infos + 1) / 2;
            int ii = 0, pp = 0;
            for (int k = 0; k < size; ++k) {
                const auto v = delta[k];
                if (v == 1)
                    continue;
                const int length = v > 9 ? nibble_count(v) : 0;
                write_nibble(info, ii++, v <= 9 ? v + 6 : length - 1);
                for (int digit = 0; digit < length; ++digit)
                    write_nibble(data, pp++, v >> (4 * digit));
            }
        }
        bytes += 4 + (infos + 1) / 2 + (payload + 1) / 2;
    }
    a.byte_counts[group] = bytes;
}

// Decode one 32-entry block cooperatively. Each lane obtains one source ID;
// callers broadcast these register values while traversing the same list block.
__device__ uint32_t decode_block(const uint8_t *data, int size, int lane, uint32_t &previous,
                                 int &bytes) {
    const uint32_t unity = uint32_t(data[0]) | (uint32_t(data[1]) << 8) |
                           (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
    const bool other = lane < size && !(unity & (1u << lane));
    const uint32_t other_mask = __ballot_sync(full_warp, other);
    const int rank = __popc(other_mask & ((1u << lane) - 1));
    const int tag = other ? read_nibble(data + 4, rank) : 0;
    const int length = other && tag < 8 ? tag + 1 : 0;
    int prefix = length;
    for (int step = 1; step < 32; step *= 2) {
        const int v = __shfl_up_sync(full_warp, prefix, step);
        if (lane >= step)
            prefix += v;
    }
    const int infos = __popc(other_mask);
    const auto *payload = data + 4 + (infos + 1) / 2;
    uint32_t delta = lane < size ? 1u : 0u;
    if (other) {
        delta = tag >= 8 ? uint32_t(tag - 6) : 0u;
        for (int d = 0; d < length; ++d)
            delta |= uint32_t(read_nibble(payload, prefix - length + d)) << (4 * d);
    }
    for (int step = 1; step < 32; step *= 2) {
        const auto v = __shfl_up_sync(full_warp, delta, step);
        if (lane >= step)
            delta += v;
    }
    const uint32_t id = previous + delta - 1;
    previous += __shfl_sync(full_warp, delta, 31);
    bytes = 4 + (infos + 1) / 2 + (__shfl_sync(full_warp, prefix, 31) + 1) / 2;
    return id;
}

// Standalone codec verification entry point. It is never used by interpolation.
__global__ void decode_kernel(ClusterCase a) {
    const int group = blockIdx.x, lane = threadIdx.x;
    uint32_t previous = 0;
    int cursor = 0;
    for (int64_t base = a.offsets[group]; base < a.offsets[group + 1]; base += 32) {
        const int size = int(min(int64_t(32), a.offsets[group + 1] - base));
        int bytes = 0;
        const auto id = decode_block(a.compressed + a.byte_offsets[group] + cursor, size, lane,
                                     previous, bytes);
        if (lane < size)
            static_cast<uint32_t *>(a.output)[base + lane] = id;
        cursor += bytes;
    }
}

template <class T, int D> __global__ void prepare_kernel(ClusterCase a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.particles)
        return;
    const auto *x = static_cast<const T *>(a.positions);
    const auto *v = static_cast<const T *>(a.values);
    auto *packed = static_cast<T *>(a.packed);
    const T volume = static_cast<const T *>(a.masses)[i] / static_cast<const T *>(a.densities)[i];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        packed[d * a.particles + i] = x[i * D + d];
        packed[(D + d) * a.particles + i] = v[i * D + d] * volume;
    }
}

// Each warp owns one small query tile (two warps for an 8x8 tile). The three
// modes differ only in source reuse: register broadcasts, warp-local shared
// memory, or one query per thread with a serial source loop and broadcasts.
template <class T, int D, int I, int J, int Mode, bool Compressed, int Warps>
__global__ void interpolate_kernel(ClusterCase a) {
    __shared__ T shared_source[Warps][2 * D][J];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    constexpr int parts = Mode == 2 ? 1 : (I * J + 31) / 32;
    const int tile = (blockIdx.x * Warps + warp) / parts;
    const int part = (blockIdx.x * Warps + warp) % parts;
    if (tile >= a.queries)
        return; // Whole warps leave together.
    const int row = Mode == 2 ? lane : part * (32 / J) + lane / J;
    const int column = lane % J;
    const int i = a.query_begin[tile] + row;
    const bool active = row < I && i < a.query_end[tile];
    const auto *x = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    T point[D], sum[D] = {};
#pragma unroll
    for (int d = 0; d < D; ++d)
        point[d] = active ? x[i * D + d] : T(0);
    const Wendland2Primal<T, D> kernel(active ? static_cast<const T *>(a.supports)[i] : T(1));
    const T cutoff = kernel.h * kernel.h * (T(1) + T(4) * machine_epsilon<T>);
    const int group = a.query_group[tile], bit = a.query_bit[tile];
    uint32_t previous = 0;
    int byte_cursor = 0;
    for (int64_t base = a.offsets[group]; base < a.offsets[group + 1]; base += 32) {
        const int size = int(min(int64_t(32), a.offsets[group + 1] - base));
        uint32_t id = 0, code = 0, mask = 0;
        if constexpr (Compressed) {
            int bytes = 0;
            id = decode_block(a.compressed + a.byte_offsets[group] + byte_cursor, size, lane,
                              previous, bytes);
            byte_cursor += bytes;
            if (lane < size)
                code = a.image_codes[base + lane];
        } else {
            const uint32_t entry = lane < size ? a.entries[base + lane] : 0;
            id = entry >> 5;
            code = entry & 31;
        }
        if (lane < size)
            mask = a.masks[base + lane];
        for (int k = 0; k < size; ++k) {
            const auto source = __shfl_sync(full_warp, id, k);
            int image_code = int(__shfl_sync(full_warp, code, k));
            if (!(__shfl_sync(full_warp, mask, k) & (1u << bit)))
                continue;
            T image[D];
#pragma unroll
            for (int d = 0; d < D; ++d) {
                image[d] = T(image_code % 3 - 1);
                image_code /= 3;
            }
            const int begin = a.source_begin[source], end = a.source_end[source];
            // First J lanes perform coalesced source loads. Remaining rows use
            // shuffles or shared memory rather than issuing duplicate loads.
            T source_data[2 * D];
#pragma unroll
            for (int d = 0; d < 2 * D; ++d) {
                source_data[d] =
                    lane < J && begin + lane < end ? packed[d * a.particles + begin + lane] : T(0);
                if constexpr (Mode == 1) {
                    if (lane < J)
                        shared_source[warp][d][lane] = source_data[d];
                }
            }
            if constexpr (Mode == 1)
                __syncwarp();
            constexpr int steps = Mode == 2 ? J : 1;
#pragma unroll
            for (int step = 0; step < steps; ++step) {
                const int j = Mode == 2 ? step : column;
                T data[2 * D];
#pragma unroll
                for (int d = 0; d < 2 * D; ++d) {
                    if constexpr (Mode == 1)
                        data[d] = shared_source[warp][d][j];
                    else
                        data[d] = __shfl_sync(full_warp, source_data[d], j);
                }
                if (active && begin + j < end) {
                    T squared = 0;
#pragma unroll
                    for (int d = 0; d < D; ++d) {
                        const T delta = (point[d] - data[d]) - image[d];
                        squared += delta * delta;
                    }
                    if (squared <= cutoff) {
                        const T weight = kernel.value(scalar_sqrt(squared));
#pragma unroll
                        for (int d = 0; d < D; ++d)
                            sum[d] += weight * data[D + d];
                    }
                }
            }
            if constexpr (Mode == 1)
                __syncwarp();
        }
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        if constexpr (Mode != 2) {
            for (int step = J / 2; step > 0; step /= 2)
                sum[d] += __shfl_down_sync(full_warp, sum[d], step, J);
        }
        if (active && (Mode == 2 || column == 0))
            static_cast<T *>(a.output)[i * D + d] = sum[d];
    }
}

// A whole query super-cluster owns a CTA. Stage 32 neighboring clusters at a
// time, sharing each source load across up to eight query-cluster warps. This
// keeps synchronization proportional to list blocks rather than source clusters.
template <class T, int D, int I, int J, bool Compressed>
__global__ void super_interpolate_kernel(ClusterCase a) {
    constexpr int parts = (I * J + 31) / 32;
    __shared__ T reference[2 * D][32 * J];
    __shared__ uint32_t source_ids[32];
    __shared__ int source_sizes[32], source_begins[32], codes[32];
    __shared__ uint8_t masks[32];
    const int group = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int query_bit = warp / parts, part = warp % parts;
    const int tile = a.group_first[group] + query_bit;
    const int row = part * (32 / J) + lane / J, column = lane % J;
    const bool valid_tile = tile < a.group_end[group];
    const int i = valid_tile ? a.query_begin[tile] + row : 0;
    const bool active = valid_tile && row < I && i < a.query_end[tile];
    const auto *x = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    T point[D], sum[D] = {};
#pragma unroll
    for (int d = 0; d < D; ++d)
        point[d] = active ? x[i * D + d] : T(0);
    const Wendland2Primal<T, D> kernel(active ? static_cast<const T *>(a.supports)[i] : T(1));
    const T cutoff = kernel.h * kernel.h * (T(1) + T(4) * machine_epsilon<T>);
    uint32_t previous = 0;
    int byte_cursor = 0;
    for (int64_t base = a.offsets[group]; base < a.offsets[group + 1]; base += 32) {
        const int size = int(min(int64_t(32), a.offsets[group + 1] - base));
        if (warp == 0) {
            uint32_t id = 0, code = 0;
            if constexpr (Compressed) {
                int bytes = 0;
                id = decode_block(a.compressed + a.byte_offsets[group] + byte_cursor, size, lane,
                                  previous, bytes);
                byte_cursor += bytes;
                if (lane < size)
                    code = a.image_codes[base + lane];
            } else {
                const auto entry = lane < size ? a.entries[base + lane] : 0;
                id = entry >> 5;
                code = entry & 31;
            }
            if (lane < size) {
                source_ids[lane] = id;
                source_begins[lane] = a.source_begin[id];
                source_sizes[lane] = a.source_end[id] - a.source_begin[id];
                codes[lane] = int(code);
                masks[lane] = a.masks[base + lane];
            }
        }
        __syncthreads();
        for (int slot = threadIdx.x; slot < size * J; slot += blockDim.x) {
            const int source = slot / J, j = slot % J;
            const int index = source_begins[source] + j;
#pragma unroll
            for (int d = 0; d < 2 * D; ++d)
                reference[d][slot] =
                    j < source_sizes[source] ? packed[d * a.particles + index] : T(0);
        }
        __syncthreads();
        for (int k = 0; k < size; ++k) {
            if (!(masks[k] & (1u << query_bit)))
                continue;
            int code = codes[k];
            T squared = 0;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const T delta = (point[d] - reference[d][k * J + column]) - T(code % 3 - 1);
                squared += delta * delta;
                code /= 3;
            }
            if (active && column < source_sizes[k] && squared <= cutoff) {
                const T weight = kernel.value(scalar_sqrt(squared));
#pragma unroll
                for (int d = 0; d < D; ++d)
                    sum[d] += weight * reference[D + d][k * J + column];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        for (int step = J / 2; step > 0; step /= 2)
            sum[d] += __shfl_down_sync(full_warp, sum[d], step, J);
        if (active && column == 0)
            static_cast<T *>(a.output)[i * D + d] = sum[d];
    }
}

template <class T, int D, int I, int J, int Mode, bool Compressed>
void launch(const ClusterCase &a, int warps) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    constexpr int parts = Mode == 2 ? 1 : (I * J + 31) / 32;
    const int blocks = (a.queries * parts + warps - 1) / warps;
    if (warps == 4)
        interpolate_kernel<T, D, I, J, Mode, Compressed, 4><<<blocks, 128, 0, stream>>>(a);
    else
        interpolate_kernel<T, D, I, J, Mode, Compressed, 8><<<blocks, 256, 0, stream>>>(a);
}
template <class T, int D, int I, int J>
void modes(const ClusterCase &a, int mode, int compressed, int warps) {
    if (mode == 3) {
        constexpr int threads = 8 * ((I * J + 31) / 32) * 32;
        const auto stream = static_cast<cudaStream_t>(a.stream);
        if (compressed)
            super_interpolate_kernel<T, D, I, J, true><<<a.groups, threads, 0, stream>>>(a);
        else
            super_interpolate_kernel<T, D, I, J, false><<<a.groups, threads, 0, stream>>>(a);
        return;
    }
#define MODE(M)                                                                                    \
    if (compressed)                                                                                \
        launch<T, D, I, J, M, true>(a, warps);                                                     \
    else                                                                                           \
        launch<T, D, I, J, M, false>(a, warps)
    if (mode == 0) {
        MODE(0);
    } else if (mode == 1) {
        MODE(1);
    } else {
        MODE(2);
    }
#undef MODE
}
template <class T, int D>
void interpolate(const ClusterCase &a, int mode, int compressed, int warps, bool prepare) {
    if (prepare)
        prepare_kernel<T, D>
            <<<(a.particles + 255) / 256, 256, 0, static_cast<cudaStream_t>(a.stream)>>>(a);
    if (a.query_size == 4 && a.source_size == 4)
        modes<T, D, 4, 4>(a, mode, compressed, warps);
    if (a.query_size == 4 && a.source_size == 8)
        modes<T, D, 4, 8>(a, mode, compressed, warps);
    if (a.query_size == 8 && a.source_size == 4)
        modes<T, D, 8, 4>(a, mode, compressed, warps);
    if (a.query_size == 8 && a.source_size == 8)
        modes<T, D, 8, 8>(a, mode, compressed, warps);
}
} // namespace cluster_detail

#define CLUSTER_DIMENSION(F, T)                                                                    \
    switch (a->dimension) {                                                                        \
    case 1:                                                                                        \
        F(T, 1);                                                                                   \
        break;                                                                                     \
    case 2:                                                                                        \
        F(T, 2);                                                                                   \
        break;                                                                                     \
    case 3:                                                                                        \
        F(T, 3);                                                                                   \
        break;                                                                                     \
    }
#define CLUSTER_DISPATCH(F)                                                                        \
    if (a->precision == 0) {                                                                       \
        CLUSTER_DIMENSION(F, float);                                                               \
    } else {                                                                                       \
        CLUSTER_DIMENSION(F, double);                                                              \
    }
EXPORT int cluster_bounds(const ClusterCase *a) {
#define RUN(T, D)                                                                                  \
    cluster_detail::bounds_kernel<T, D>                                                            \
        <<<(a->queries + 255) / 256, 256, 0, static_cast<cudaStream_t>(a->stream)>>>(*a, false);   \
    cluster_detail::bounds_kernel<T, D>                                                            \
        <<<(a->sources + 255) / 256, 256, 0, static_cast<cudaStream_t>(a->stream)>>>(*a, true)
    CLUSTER_DISPATCH(RUN);
#undef RUN
    return int(cudaGetLastError());
}
EXPORT int cluster_build(const ClusterCase *a, int fill, int exact) {
#define RUN(T, D)                                                                                  \
    if (fill)                                                                                      \
        cluster_detail::build_kernel<T, D, true>                                                   \
            <<<a->groups, 128, 0, static_cast<cudaStream_t>(a->stream)>>>(*a, exact);              \
    else                                                                                           \
        cluster_detail::build_kernel<T, D, false>                                                  \
            <<<a->groups, 128, 0, static_cast<cudaStream_t>(a->stream)>>>(*a, exact)
    CLUSTER_DISPATCH(RUN);
#undef RUN
    return int(cudaGetLastError());
}
EXPORT int cluster_compress(const ClusterCase *a, int fill) {
    const auto stream = static_cast<cudaStream_t>(a->stream);
    if (fill)
        cluster_detail::compress_kernel<true><<<(a->groups + 127) / 128, 128, 0, stream>>>(*a);
    else
        cluster_detail::compress_kernel<false><<<(a->groups + 127) / 128, 128, 0, stream>>>(*a);
    return int(cudaGetLastError());
}
EXPORT int cluster_decode(const ClusterCase *a) {
    cluster_detail::decode_kernel<<<a->groups, 32, 0, static_cast<cudaStream_t>(a->stream)>>>(*a);
    return int(cudaGetLastError());
}
EXPORT int cluster_interpolate(const ClusterCase *a, int mode, int compressed, int warps,
                               int prepare, int iterations) {
    for (int i = 0; i < iterations; ++i) {
#define RUN(T, D) cluster_detail::interpolate<T, D>(*a, mode, compressed, warps, prepare)
        CLUSTER_DISPATCH(RUN);
#undef RUN
    }
    return int(cudaGetLastError());
}
