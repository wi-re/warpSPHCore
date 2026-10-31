#pragma once
#include "execution/packedReference.cuh"
#include "kernels/kernel.h"
#include "math/queryGeometry.h"
namespace sph_native {
// Aligned temporary records put a reference position beside the mass/weighted
// field consumed by density/interpolation. Preparation is O(reference count),
// amortized over CSR edges; no persistent cache or changed user tensor layout.
template <int D, int Components, int Width, bool Wide, class T = float, int Batch = 1>
struct PackedValueRow {
    RowArgs<T, D> a;
    const T *packed;
    __device__ void operator()(int64_t i, int lane) const {
        constexpr int stride = packed_reference_stride<D, Components>;
        if constexpr (Batch == 1) {
            T sum[Components] = {};
            const QueryGeometry<T, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
            const Wendland2Primal<T, D> kernel(a.qh[i]);
            const int64_t begin = a.offsets[i];
            for (int edge = lane; edge < a.counts[i]; edge += Width) {
                const int j = packed_reference_index<Wide>(a.neighbors, begin + edge);
                T record[stride];
                load_packed_reference(packed, j, record);
                T dx[D], position[D];
#pragma unroll
                for (int d = 0; d < D; ++d)
                    position[d] = record[d];
                const T w = kernel.value(geometry.distance(position, dx));
#pragma unroll
                for (int c = 0; c < Components; ++c)
                    sum[c] += record[D + c] * w;
            }
#pragma unroll
            for (int c = 0; c < Components; ++c) {
                const T result = group_sum<Width>(sum[c]);
                if (lane == 0)
                    a.output[i * Components + c] = result;
            }
        } else {
            T sums[Batch][Components] = {};
            const QueryGeometry<T, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
            const Wendland2Primal<T, D> kernel(a.qh[i]);
            const int64_t begin = a.offsets[i];
            const int count = a.counts[i];
            for (int base = lane; base < count; base += Width * Batch) {
                int indices[Batch];
                T records[Batch][stride];
                // Expose independent address chains before waiting for a record.
                // Batching is intentionally small: larger batches reduce occupancy.
#pragma unroll
                for (int u = 0; u < Batch; ++u)
                    indices[u] =
                        base + u * Width < count
                            ? packed_reference_index<Wide>(a.neighbors, begin + base + u * Width)
                            : 0;
#pragma unroll
                for (int u = 0; u < Batch; ++u)
                    if (base + u * Width < count)
                        load_packed_reference(packed, indices[u], records[u]);
#pragma unroll
                for (int u = 0; u < Batch; ++u) {
                    if (base + u * Width < count) {
                        T dx[D];
                        const T w = kernel.value(geometry.distance(records[u], dx));
#pragma unroll
                        for (int c = 0; c < Components; ++c)
                            sums[u][c] += records[u][D + c] * w;
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < Components; ++c) {
                T sum = 0;
#pragma unroll
                for (int u = 0; u < Batch; ++u)
                    sum += sums[u][c];
                const T result = group_sum<Width>(sum);
                if (lane == 0)
                    a.output[i * Components + c] = result;
            }
        }
    }
};

template <bool Density, class T = float, int Batch = 1> struct PackedValuesRows {
    T *packed;
    template <int D, int Components, class Launch>
    void fields(int64_t n, RowArgs<T, D> a, const Launch &launch) const {
        CooperativeGpuLauncher<1>{launch.stream}(
            a.references,
            PreparePackedReference<D, Components,
                                   Density ? PackedContent::Mass : PackedContent::WeightedField, T>{
                a, packed});
        if (a.neighbors.wide)
            launch(n, PackedValueRow<D, Components, Launch::width, true, T, Batch>{a, packed});
        else
            launch(n, PackedValueRow<D, Components, Launch::width, false, T, Batch>{a, packed});
    }
    template <int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch &launch) const {
        if constexpr (Density)
            fields<D, 1>(n, a, launch);
        else if (a.fields == 1)
            fields<D, 1>(n, a, launch);
        else if (a.fields == D)
            fields<D, D>(n, a, launch);
        else
            fields<D, D * D>(n, a, launch);
    }
};
} // namespace sph_native
