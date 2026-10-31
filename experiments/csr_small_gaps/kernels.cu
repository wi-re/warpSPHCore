// Exact full-neighbor-list codecs. This experiment borrows the immutable byte
// experiment's ABI, packing, geometry and raw controls, but adds no production
// dispatch or AD paths. Compressed consumers retain only counts and row streams.
#include "../csr_compression/kernels.cu"

namespace small_gaps {
using namespace compressed_csr;
enum Format { Band = 4, Nibble = 5 };

__device__ uint32_t read_word(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
__device__ void write_word(uint8_t *p, uint32_t value) {
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = value >> (8 * i);
}
__device__ unsigned digits(uint32_t gap) { return (32 - __clz(gap) + 3) / 4; }
__device__ unsigned get_nibble(const uint8_t *p, unsigned index) {
    return (p[index / 2] >> (4 * (index & 1))) & 15;
}
// Only the offline encoder uses read/modify/write: one thread owns an entire
// row, so adjacent nibbles never race. The consumer only performs reads.
__device__ void put_nibble(uint8_t *p, unsigned index, unsigned value) {
    if (index & 1) p[index / 2] |= uint8_t(value << 4);
    else p[index / 2] = uint8_t(value);
}

// Two-pass planning. CSR exists only here and in the verification kernels.
// Rows start on four-byte boundaries; offsets are 64-bit even past 4 GiB.
template <int Encoding, int Chunk, bool Fill>
__global__ void encode_rows(CompressionCase a) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= a.queries) return;
    const int count = a.counts[row];
    const auto *ids = a.indices + a.row_offsets[row];
    uint8_t *stream = Fill ? a.codes + a.row_blocks[row] : nullptr;
    if (!count) {
        if constexpr (!Fill) a.meta_counts[row] = 0;
        return;
    }
    if constexpr (Fill) write_word(stream, uint32_t(ids[0]));
    unsigned bytes = 4;
    if constexpr (Encoding == Band) {
        const unsigned controls = (unsigned(count) + 3) / 4;
        bytes += controls;
        unsigned control = 0;
        for (int edge = 0; edge < count; ++edge) {
            const uint32_t gap = edge ? uint32_t(ids[edge] - ids[edge - 1]) : 0;
            const unsigned tag = gap <= 1 ? 0 : gap == 2 ? 1 : gap <= 256 ? 2 : 3;
            const unsigned length = tag == 2 ? 1 : tag == 3 ? 4 : 0;
            if constexpr (Fill) {
                control |= tag << (2 * (edge & 3));
                if ((edge & 3) == 3 || edge + 1 == count) {
                    stream[4 + edge / 4] = control;
                    control = 0;
                }
                const uint32_t shifted = gap - 1;
                for (unsigned k = 0; k < length; ++k) stream[bytes + k] = shifted >> (8 * k);
            }
            bytes += length;
        }
    } else {
        for (int start = 0; start < count; start += Chunk) {
            unsigned nonunit = 0, payload = 0;
            uint32_t unity = 0xffffffffu;
            for (int lane = 0; lane < Chunk && start + lane < count; ++lane) {
                const int edge = start + lane;
                const uint32_t gap = edge ? uint32_t(ids[edge] - ids[edge - 1]) : 1;
                if (gap != 1) { unity &= ~(1u << lane); ++nonunit; }
                if (gap > 9) payload += digits(gap);
            }
            if constexpr (Fill) {
                write_word(stream + bytes, unity);
                auto *info = stream + bytes + 4;
                auto *data = info + (nonunit + 1) / 2;
                unsigned info_cursor = 0, data_cursor = 0;
                for (int lane = 0; lane < Chunk && start + lane < count; ++lane) {
                    const int edge = start + lane;
                    const uint32_t gap = edge ? uint32_t(ids[edge] - ids[edge - 1]) : 1;
                    if (gap == 1) continue;
                    const unsigned length = gap > 9 ? digits(gap) : 0;
                    put_nibble(info, info_cursor++, length ? length - 1 : gap + 6);
                    for (unsigned k = 0; k < length; ++k)
                        put_nibble(data, data_cursor++, (gap >> (4 * k)) & 15);
                }
            }
            bytes += 4 + (nonunit + 1) / 2 + (payload + 1) / 2;
        }
    }
    if constexpr (!Fill) a.meta_counts[row] = (bytes + 3) & ~3u;
}

// Each subgroup carries the previous ID and payload cursor across the row.
// Items adjacent edges per lane reduce scan/shuffle count, at the cost of more
// live IDs and records. No neighbor ID is materialized in global memory.
template <int Encoding, int Chunk, int Items> struct Reader {
    static constexpr int Width = Chunk / Items;
    const uint8_t *row;
    uint64_t cursor;
    uint32_t previous;
    int count;
    __device__ Reader(const CompressionCase &a, int query)
        : row(nullptr), cursor(4), previous(0), count(a.counts[query]) {
        if constexpr (Encoding >= Band) {
            row = a.codes + a.row_blocks[query];
            if (count) previous = read_word(row);
        }
        if constexpr (Encoding == Band) cursor += (unsigned(count) + 3) / 4;
    }
    __device__ void next(int start, int lane, int (&ids)[Items]) {
        unsigned tags[Items], lengths[Items], local_length = 0;
        unsigned nonunit = 0;
        const uint8_t *data;
        if constexpr (Encoding == Nibble) {
            nonunit = ~read_word(row + cursor);
            if constexpr (Chunk < 32) nonunit &= (1u << Chunk) - 1;
            data = row + cursor + 4 + (__popc(nonunit) + 1) / 2;
        } else data = row + cursor;
#pragma unroll
        for (int u = 0; u < Items; ++u) {
            const int position = lane * Items + u, edge = start + position;
            if constexpr (Encoding == Band) {
                tags[u] = edge < count ? (row[4 + edge / 4] >> (2 * (edge & 3))) & 3 : 0;
                lengths[u] = tags[u] == 2 ? 1 : tags[u] == 3 ? 4 : 0;
            } else {
                const bool extra = nonunit & (1u << position);
                const unsigned rank = __popc(nonunit & lower_lanes(position));
                tags[u] = extra ? get_nibble(row + cursor + 4, rank) : 0;
                lengths[u] = extra && tags[u] < 8 ? tags[u] + 1 : 0;
            }
            local_length += lengths[u];
        }
        const unsigned end = inclusive_sum<Width>(local_length);
        unsigned payload = end - local_length;
        unsigned total = 0, prefix[Items];
#pragma unroll
        for (int u = 0; u < Items; ++u) {
            const int position = lane * Items + u, edge = start + position;
            uint32_t gap;
            if constexpr (Encoding == Band) {
                uint32_t value = 0;
                for (unsigned k = 0; k < lengths[u]; ++k) value |= uint32_t(data[payload + k]) << (8 * k);
                gap = tags[u] < 2 ? tags[u] + 1 : value + 1;
            } else {
                uint32_t value = 0;
                for (unsigned k = 0; k < lengths[u]; ++k)
                    value |= get_nibble(data, payload + k) << (4 * k);
                gap = !(nonunit & (1u << position)) ? 1 : lengths[u] ? value : tags[u] - 6;
            }
            payload += lengths[u];
            // Edge zero is the stored absolute ID. Padding must contribute zero
            // to the carry, especially for singleton and incomplete final chunks.
            total += edge && edge < count ? gap : 0;
            prefix[u] = total;
        }
        const uint32_t inclusive = inclusive_sum<Width>(total);
#pragma unroll
        for (int u = 0; u < Items; ++u) ids[u] = int(previous + inclusive - total + prefix[u]);
        previous += __shfl_sync(subgroup_mask<Width>(), inclusive, Width - 1, Width);
        const unsigned consumed = __shfl_sync(subgroup_mask<Width>(), end, Width - 1, Width);
        if constexpr (Encoding == Band) cursor += consumed;
        else cursor += 4 + (__popc(nonunit) + 1) / 2 + (consumed + 1) / 2;
    }
};

template <int Encoding, int Chunk, int Items>
__global__ void decode_rows(CompressionCase a) {
    constexpr int Width = Chunk / Items;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (row >= a.queries) return;
    Reader<Encoding, Chunk, Items> reader(a, row);
    for (int start = 0; start < a.counts[row]; start += Chunk) {
        int ids[Items];
        reader.next(start, lane, ids);
#pragma unroll
        for (int u = 0; u < Items; ++u)
            if (start + lane * Items + u < a.counts[row])
                a.decoded[a.row_offsets[row] + start + lane * Items + u] = ids[u];
    }
}

template <class T, int D, int Encoding, int Chunk, int Items, int Batch>
__global__ void interpolate(CompressionCase a) {
    constexpr int Width = Chunk / Items, Slots = Items * Batch;
    constexpr int Pitch = ((2 * D + 3) / 4) * 4;
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (query >= a.queries) return;
    const auto *x = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    const QueryGeometry<T, D> geometry(x + int64_t(query) * D, static_cast<const T *>(a.lower),
                                     static_cast<const T *>(a.upper), a.periodic);
    const Wendland2Primal<T, D> kernel(static_cast<const T *>(a.supports)[query]);
    const int count = a.counts[query];
    // Raw controls use this exact schedule, including accumulation order.
    Reader<Encoding, Chunk, Items> reader(a, query);
    T sums[Slots][D] = {};
    for (int start = 0; start < count; start += Chunk * Batch) {
        int ids[Batch][Items];
        T records[Slots][Pitch];
#pragma unroll
        for (int b = 0; b < Batch; ++b) {
            if constexpr (Encoding < Band) {
#pragma unroll
                for (int u = 0; u < Items; ++u) {
                    const int edge = start + b * Chunk + lane * Items + u;
                    if (edge < count) {
                        const int64_t offset = a.row_offsets[query] + edge;
                        ids[b][u] = Encoding == Raw64 ? int(a.indices[offset])
                                          : reinterpret_cast<const int32_t *>(a.indices)[offset];
                    }
                }
            } else if (start + b * Chunk < count) reader.next(start + b * Chunk, lane, ids[b]);
        }
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int edge = start + (slot / Items) * Chunk + lane * Items + slot % Items;
            if (edge < count) load_packed_reference(packed, ids[slot / Items][slot % Items], records[slot]);
        }
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int edge = start + (slot / Items) * Chunk + lane * Items + slot % Items;
            if (edge < count) {
                T dx[D];
                const T weight = kernel.value(geometry.distance(records[slot], dx));
#pragma unroll
                for (int d = 0; d < D; ++d) sums[slot][d] += records[slot][D + d] * weight;
            }
        }
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        T sum = 0;
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) sum += sums[slot][d];
        const T total = group_sum<Width>(sum);
        if (!lane) static_cast<T *>(a.output)[int64_t(query) * D + d] = total;
    }
}

template <int Encoding, int Chunk, int Items>
void codec(const CompressionCase &a, int action) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (!a.queries) return;
    if (action == 0) encode_rows<Encoding, Chunk, false><<<(a.queries + 127) / 128, 128, 0, stream>>>(a);
    else if (action == 1) encode_rows<Encoding, Chunk, true><<<(a.queries + 127) / 128, 128, 0, stream>>>(a);
    else decode_rows<Encoding, Chunk, Items><<<(int64_t(a.queries) * (Chunk / Items) + 127) / 128, 128, 0, stream>>>(a);
}
template <class T, int D, int Encoding, int Chunk, int Items, int Batch>
void launch(const CompressionCase &a, int threads, bool pack) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (pack && a.references) prepare<T, D><<<(a.references + 255) / 256, 256, 0, stream>>>(a);
    if (a.queries) interpolate<T,D,Encoding,Chunk,Items,Batch>
        <<<(int64_t(a.queries) * (Chunk / Items) + threads - 1) / threads, threads, 0, stream>>>(a);
}
template <int Encoding, int Chunk, int Items>
void run(const CompressionCase &a, int batch, int threads, bool pack) {
#define LAUNCH(T,D,B) launch<T,D,Encoding,Chunk,Items,B>(a,threads,pack)
#define TYPE(T,D) if (batch == 1) { LAUNCH(T,D,1); } else { LAUNCH(T,D,2); }
    if (a.precision) {
        if (a.dimension == 1) { TYPE(double,1); }
        else if (a.dimension == 2) { TYPE(double,2); }
        else { TYPE(double,3); }
    } else {
        if (a.dimension == 1) { TYPE(float,1); }
        else if (a.dimension == 2) { TYPE(float,2); }
        else { TYPE(float,3); }
    }
#undef TYPE
#undef LAUNCH
}
} // namespace small_gaps

// Dispatch is deliberately outside the hot loop. Arguments are private to the
// experiment, rather than additional knobs on production SPH operations.
#define ITEMS(F,C) if (items == 1) { F(C,1); } else if (items == 2) { F(C,2); } else { F(C,4); }
#define CHUNKS(F) switch (chunk) { case 4: ITEMS(F,4); break; case 8: ITEMS(F,8); break; case 16: ITEMS(F,16); break; case 32: ITEMS(F,32); break; }
extern "C" int gaps_codec(const CompressionCase *a, int format, int chunk, int items, int action) {
#define CODEC(C,I) if (format == small_gaps::Band) small_gaps::codec<small_gaps::Band,C,I>(*a,action); else small_gaps::codec<small_gaps::Nibble,C,I>(*a,action)
    CHUNKS(CODEC);
#undef CODEC
    return int(cudaGetLastError());
}
extern "C" int gaps_interpolate(const CompressionCase *a, int format, int chunk, int items, int batch,
                                 int threads, int pack, int iterations) {
    for (int repeat = 0; repeat < iterations; ++repeat) {
#define RUN(C,I) switch (format) { case 0: small_gaps::run<0,C,I>(*a,batch,threads,pack); break; case 1: small_gaps::run<1,C,I>(*a,batch,threads,pack); break; case 4: small_gaps::run<4,C,I>(*a,batch,threads,pack); break; case 5: small_gaps::run<5,C,I>(*a,batch,threads,pack); break; }
        CHUNKS(RUN);
#undef RUN
    }
    return int(cudaGetLastError());
}
#undef CHUNKS
#undef ITEMS
