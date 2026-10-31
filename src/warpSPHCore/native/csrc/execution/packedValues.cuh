#pragma once
#include "execution/packedReference.cuh"
#include "kernels/kernel.h"
#include "math/queryGeometry.h"
namespace sph_native {
// Aligned temporary records put a reference position beside the mass/weighted
// field consumed by density/interpolation. Preparation is O(reference count),
// amortized over CSR edges; no persistent cache or changed user tensor layout.
template <int D, int Components, int Width, bool Wide> struct PackedValueRow {
    RowArgs<float, D> a;
    const float *packed;
    __device__ void operator()(int64_t i, int lane) const {
        constexpr int stride = packed_reference_stride<D, Components>;
        float sum[Components] = {};
        const QueryGeometry<float, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
        const Wendland2Primal<float, D> kernel(a.qh[i]);
        const int64_t begin = a.offsets[i];
        for (int edge = lane; edge < a.counts[i]; edge += Width) {
            const int j = packed_reference_index<Wide>(a.neighbors, begin + edge);
            float record[stride];
            load_packed_reference(packed, j, record);
            float dx[D], position[D];
#pragma unroll
            for (int d = 0; d < D; ++d)
                position[d] = record[d];
            const float w = kernel.value(geometry.distance(position, dx));
#pragma unroll
            for (int c = 0; c < Components; ++c)
                sum[c] += record[D + c] * w;
        }
#pragma unroll
        for (int c = 0; c < Components; ++c) {
            const float result = group_sum<Width>(sum[c]);
            if (lane == 0)
                a.output[i * Components + c] = result;
        }
    }
};

template <bool Density> struct PackedValuesRows {
    float *packed;
    template <int D, int Components, class Launch>
    void fields(int64_t n, RowArgs<float, D> a, const Launch &launch) const {
        CooperativeGpuLauncher<1>{launch.stream}(
            a.references,
            PreparePackedReference<D, Components,
                                   Density ? PackedContent::Mass : PackedContent::WeightedField>{
                a, packed});
        if (a.neighbors.wide)
            launch(n, PackedValueRow<D, Components, Launch::width, true>{a, packed});
        else
            launch(n, PackedValueRow<D, Components, Launch::width, false>{a, packed});
    }
    template <int D, class Launch>
    void operator()(int64_t n, RowArgs<float, D> a, const Launch &launch) const {
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
