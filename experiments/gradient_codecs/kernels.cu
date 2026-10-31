// Heavier primal consumers of the existing exact CSR codecs. Production kernels
// and AD are untouched. The included experiments provide the same ABI, lossless
// readers, packed loads and numerical geometry used by the interpolation study.
#include "../csr_small_gaps/kernels.cu"

struct GradientCase {
    CompressionCase pairs;
    const void *volumes, *A, *B, *gradA, *gradB;
};

namespace gradient_codecs {
using namespace compressed_csr;
using small_gaps::Band;
using small_gaps::Nibble;
// Scalar gradient, vector Jacobian, vector divergence, and covariance matrix.
template <int D, int Work> constexpr int fields = Work == 0 ? 1 : Work == 3 ? 0 : D;
template <int D, int Work> constexpr int outputs = Work == 0 ? D : Work == 2 ? 1 : D * D;
template <int D, int Work> constexpr int pitch = ((D + 1 + fields<D, Work> + 3) / 4) * 4;

template <class T, int D, int Work, bool Corrected>
__global__ void prepare(GradientCase request) {
    const auto &a = request.pairs;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= a.references) return;
    T *record = static_cast<T *>(a.packed) + int64_t(j) * pitch<D, Work>;
#pragma unroll
    for (int d = 0; d < D; ++d) record[d] = static_cast<const T *>(a.positions)[int64_t(j) * D + d];
    if constexpr (Corrected) record[D] = static_cast<const T *>(request.volumes)[j];
    else record[D] = static_cast<const T *>(a.masses)[j] / static_cast<const T *>(a.densities)[j];
#pragma unroll
    for (int c = 0; c < fields<D, Work>; ++c)
        record[D + 1 + c] = static_cast<const T *>(a.values)[int64_t(j) * fields<D, Work> + c];
#pragma unroll
    for (int c = D + 1 + fields<D, Work>; c < pitch<D, Work>; ++c) record[c] = 0;
}

// Adjacent codes per lane, using the previous byte format's independent chunk
// directory. This is the same lossless prefix sum as its scalar/four-code reader.
template <int Chunk, int Items>
__device__ void byte_ids(const CompressionCase &a, int query, int start, int lane, int (&ids)[Items]) {
    constexpr int Width = Chunk / Items;
    const int64_t block = a.row_blocks[query] + start / Chunk;
    unsigned codes[Items], escapes = 0;
#pragma unroll
    for (int u = 0; u < Items; ++u) {
        codes[u] = a.codes[block * Chunk + lane * Items + u];
        escapes += codes[u] == 255;
    }
    const unsigned end = inclusive_sum<Width>(escapes);
    unsigned cursor = a.meta_offsets[block] + end - escapes;
    uint32_t local[Items], total = 0;
#pragma unroll
    for (int u = 0; u < Items; ++u) {
        total += codes[u] == 255 ? a.extras[cursor++] : codes[u];
        local[u] = total;
    }
    const unsigned before = inclusive_sum<Width>(total) - total;
#pragma unroll
    for (int u = 0; u < Items; ++u) ids[u] = int(a.bases[block] + before + local[u]);
}

template <class T, int D, int Work, bool Corrected, int Encoding, int Chunk, int Items, int Batch>
__global__ void consume(GradientCase request) {
    constexpr int Width = Chunk / Items, Slots = Items * Batch;
    constexpr int Fields = fields<D, Work>, Outputs = outputs<D, Work>, Pitch = pitch<D, Work>;
    const auto &a = request.pairs;
    const int query = (blockIdx.x * blockDim.x + threadIdx.x) / Width;
    const int lane = threadIdx.x & (Width - 1);
    if (query >= a.queries) return;
    const auto *x = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    const QueryGeometry<T, D> geometry(x + int64_t(query) * D, static_cast<const T *>(a.lower),
                                     static_cast<const T *>(a.upper), a.periodic);
    const Wendland2Primal<T, D> kernel(static_cast<const T *>(a.supports)[query]);
    T query_field[Fields ? Fields : 1], A = 0, B[D], gradA[D], gradB[D][D];
#pragma unroll
    for (int c = 0; c < Fields; ++c) query_field[c] = static_cast<const T *>(a.values)[int64_t(query) * Fields + c];
    if constexpr (Corrected) {
        A = static_cast<const T *>(request.A)[query];
#pragma unroll
        for (int d = 0; d < D; ++d) {
            B[d] = static_cast<const T *>(request.B)[int64_t(query) * D + d];
            gradA[d] = static_cast<const T *>(request.gradA)[int64_t(query) * D + d];
#pragma unroll
            for (int c = 0; c < D; ++c)
                gradB[d][c] = static_cast<const T *>(request.gradB)[(int64_t(query) * D + d) * D + c];
        }
    }
    small_gaps::Reader<Encoding, Chunk, Items> reader(a, query);
    T sums[Slots][Outputs] = {};
    const int count = a.counts[query];
    for (int start = 0; start < count; start += Chunk * Batch) {
        int ids[Batch][Items];
        T records[Slots][Pitch];
#pragma unroll
        for (int b = 0; b < Batch; ++b) {
            if constexpr (Encoding < Delta) {
#pragma unroll
                for (int u = 0; u < Items; ++u) {
                    const int edge = start + b * Chunk + lane * Items + u;
                    if (edge < count) {
                        const int64_t at = a.row_offsets[query] + edge;
                        ids[b][u] = Encoding == Raw64 ? int(a.indices[at]) : reinterpret_cast<const int32_t *>(a.indices)[at];
                    }
                }
            } else if (start + b * Chunk < count) {
                if constexpr (Encoding == Delta) byte_ids<Chunk,Items>(a,query,start + b * Chunk,lane,ids[b]);
                else reader.next(start + b * Chunk,lane,ids[b]);
            }
        }
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int edge = start + (slot / Items) * Chunk + lane * Items + slot % Items;
            if (edge < count) load_packed_reference(packed,ids[slot / Items][slot % Items],records[slot]);
        }
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int edge = start + (slot / Items) * Chunk + lane * Items + slot % Items;
            if (edge >= count) continue;
            const T *record = records[slot];
            T dx[D], w, g, gradient[D];
            const T r = geometry.distance(record,dx);
            kernel.evaluate(r,w,g);
#pragma unroll
            for (int d = 0; d < D; ++d) gradient[d] = g * dx[d];
            if constexpr (Corrected) {
                T dot = 0;
#pragma unroll
                for (int d = 0; d < D; ++d) dot += B[d] * dx[d];
                const T one_plus_dot = T(1) + dot;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    T product = 0;
#pragma unroll
                    for (int c = 0; c < D; ++c) product += gradB[d][c] * dx[c];
                    // gradB[d,c] is derivative-first. Match Warp's four terms,
                    // including the nonsymmetric matrix contraction gradB @ dx.
                    const T term1 = (A * w) * B[d];
                    const T term2 = (A * one_plus_dot) * gradient[d];
                    const T term3 = (one_plus_dot * w) * gradA[d];
                    const T term4 = (A * w) * product;
                    gradient[d] = term1 + term2 + term3 + term4;
                }
            }
            const T volume = record[D];
            if constexpr (Work == 3) {
#pragma unroll
                for (int b = 0; b < D; ++b)
#pragma unroll
                    for (int d = 0; d < D; ++d) sums[slot][b * D + d] += (-dx[b] * volume) * gradient[d];
            } else if constexpr (Work == 2) {
#pragma unroll
                for (int d = 0; d < D; ++d)
                    sums[slot][0] += ((record[D + 1 + d] - query_field[d]) * volume) * gradient[d];
            } else {
#pragma unroll
                for (int c = 0; c < Fields; ++c) {
                    const T weighted = (record[D + 1 + c] - query_field[c]) * volume;
#pragma unroll
                    for (int d = 0; d < D; ++d) sums[slot][c * D + d] += weighted * gradient[d];
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < Outputs; ++c) {
        T sum = 0;
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) sum += sums[slot][c];
        const T total = group_sum<Width>(sum);
        if (!lane) static_cast<T *>(a.output)[int64_t(query) * Outputs + c] = total;
    }
}

template <class T, int D, int Work, bool Corrected, int Encoding, int Chunk, int Items, int Batch>
void launch(const GradientCase &a, int threads, bool pack) {
    const auto stream = static_cast<cudaStream_t>(a.pairs.stream);
    if (pack && a.pairs.references)
        prepare<T,D,Work,Corrected><<<(a.pairs.references + 255) / 256,256,0,stream>>>(a);
    if (a.pairs.queries)
        consume<T,D,Work,Corrected,Encoding,Chunk,Items,Batch>
            <<<(int64_t(a.pairs.queries) * (Chunk / Items) + threads - 1) / threads,threads,0,stream>>>(a);
}
template <int Work, bool Corrected, int Encoding, int Chunk, int Items, int Batch>
void precision(const GradientCase &a, int threads, bool pack) {
#define RUN(T,D) launch<T,D,Work,Corrected,Encoding,Chunk,Items,Batch>(a,threads,pack)
    if (a.pairs.precision) {
        if (a.pairs.dimension == 1) { RUN(double,1); } else if (a.pairs.dimension == 2) { RUN(double,2); } else { RUN(double,3); }
    } else {
        if (a.pairs.dimension == 1) { RUN(float,1); } else if (a.pairs.dimension == 2) { RUN(float,2); } else { RUN(float,3); }
    }
#undef RUN
}
template <int Encoding, int Chunk, int Items, int Batch>
void operation(const GradientCase &a, int work, bool corrected, int threads, bool pack) {
#define WORK(W) if (corrected) precision<W,true,Encoding,Chunk,Items,Batch>(a,threads,pack); else precision<W,false,Encoding,Chunk,Items,Batch>(a,threads,pack)
    switch (work) { case 0: WORK(0); break; case 1: WORK(1); break; case 2: WORK(2); break; case 3: WORK(3); break; }
#undef WORK
}
template <int Chunk, int Items, int Batch>
void format(const GradientCase &a, int encoding, int work, bool corrected, int threads, bool pack) {
#define FORMAT(F) operation<F,Chunk,Items,Batch>(a,work,corrected,threads,pack)
    switch (encoding) { case 0: FORMAT(0); break; case 1: FORMAT(1); break; case 2: FORMAT(2); break; case 4: FORMAT(4); break; case 5: FORMAT(5); break; }
#undef FORMAT
}
} // namespace gradient_codecs

extern "C" int gradient_consume(const GradientCase *a, int encoding, int work, int corrected,
                                int chunk, int items, int batch, int threads, int pack, int iterations) {
    for (int repeat = 0; repeat < iterations; ++repeat) {
#define RUN(C,I) if (batch == 1) gradient_codecs::format<C,I,1>(*a,encoding,work,corrected,threads,pack); else gradient_codecs::format<C,I,2>(*a,encoding,work,corrected,threads,pack)
#define CHUNK(C) if (items == 1) { RUN(C,1); } else { RUN(C,2); }
        switch (chunk) { case 8: CHUNK(8); break; case 16: CHUNK(16); break; case 32: CHUNK(32); break; }
#undef CHUNK
#undef RUN
    }
    return int(cudaGetLastError());
}
