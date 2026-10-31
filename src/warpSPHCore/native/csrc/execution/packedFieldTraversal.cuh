#pragma once
#include "execution/packedReference.cuh"
#include "execution/fieldTraversal.cuh"
namespace sph_native {
template <int D, int Fields, int Operation, int Width, bool Wide, class Pair>
struct PackedFieldRow {
    RowArgs<float, D> input;
    const float *packed;
    __device__ void operator()(int64_t i, int lane) const {
        constexpr int stored = Operation == 7 ? 0 : Fields;
        constexpr int stride = packed_reference_stride<D, stored + 1>;
        constexpr int outputs = field_outputs<D, Fields, Operation>();
        float query_field[Fields] = {}, sum[D * D * D] = {};
        if constexpr (Operation != 7) {
#pragma unroll
            for (int c = 0; c < Fields; ++c)
                query_field[c] = input.qvalue[i * Fields + c];
        }
        PackedFieldArgs<float, D> a{input, nullptr, query_field};
        a.fields = Fields;
        a.outputs = outputs;
        a.support = 11;
        a.scheme = 3;
        a.laplacian_mode = 2;
        const QueryGeometry<float, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
        const Wendland2Primal<float, D> kernel(a.qh[i]);
        const int64_t begin = a.offsets[i];
        for (int edge = lane; edge < a.counts[i]; edge += Width) {
            const int j = packed_reference_index<Wide>(a.neighbors, begin + edge);
            float record[stride], position[D], dx[D], w, g;
            load_packed_reference(packed, j, record);
#pragma unroll
            for (int d = 0; d < D; ++d)
                position[d] = record[d];
            const float r = geometry.distance(position, dx);
            kernel.evaluate(r, w, g);
            a.reference = record;
            Pair::accumulate(a, i, j, dx, r, w, g, sum);
        }
#pragma unroll
        for (int c = 0; c < outputs; ++c) {
            const float value = group_sum<Width>(sum[c]);
            if (lane == 0)
                a.output[i * outputs + c] = value;
        }
    }
};
template <template <class, int> class Pair, int Operation> struct PackedFieldRows {
    float *packed;
    template <int D, int Fields, class Launch>
    void fields(int64_t n, RowArgs<float, D> a, const Launch &launch) const {
        constexpr int stored = Operation == 7 ? 0 : Fields;
        CooperativeGpuLauncher<1>{launch.stream}(
            a.references,
            PreparePackedReference<D, stored, PackedContent::VolumeAndField>{a, packed});
        if (a.neighbors.wide)
            launch(n, PackedFieldRow<D, Fields, Operation, Launch::width, true, Pair<float, D>>{
                          a, packed});
        else
            launch(n, PackedFieldRow<D, Fields, Operation, Launch::width, false, Pair<float, D>>{
                          a, packed});
    }
    template <int D, class Launch>
    void operator()(int64_t n, RowArgs<float, D> a, const Launch &launch) const {
        if (a.fields == 1)
            fields<D, 1>(n, a, launch);
        else if (a.fields == D)
            fields<D, D>(n, a, launch);
        else
            fields<D, D * D>(n, a, launch);
    }
};
} // namespace sph_native
