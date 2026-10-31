#pragma once
#include "execution/cooperative.cuh"
#include "dataTypes/state.h"
namespace sph_native {
enum class PackedContent { Mass, WeightedField, VolumeAndField };
template <int D, int Components>
constexpr int packed_reference_stride = ((D + Components + 3) / 4) * 4;

// Reference-side work is performed once, then reused for every incident edge.
// Padding is initialized so vector loads never read uninitialized storage.
template <int D, int Fields, PackedContent Content> struct PreparePackedReference {
    RowArgs<float, D> a;
    float *packed;
    __device__ void operator()(int64_t j, int lane) const {
        if (lane != 0)
            return;
        constexpr int components = Fields + (Content == PackedContent::VolumeAndField ? 1 : 0);
        constexpr int stride = packed_reference_stride<D, components>;
#pragma unroll
        for (int d = 0; d < D; ++d)
            packed[j * stride + d] = a.rx[j * D + d];
        if constexpr (Content == PackedContent::Mass) {
            packed[j * stride + D] = a.mass[j];
        } else {
            const float volume = a.mass[j] / a.density[j];
            if constexpr (Content == PackedContent::VolumeAndField) {
                packed[j * stride + D] = volume;
#pragma unroll
                for (int c = 0; c < Fields; ++c)
                    packed[j * stride + D + 1 + c] = a.rvalue[j * Fields + c];
            } else {
#pragma unroll
                for (int c = 0; c < Fields; ++c)
                    packed[j * stride + D + c] = a.rvalue[j * Fields + c] * volume;
            }
        }
#pragma unroll
        for (int c = D + components; c < stride; ++c)
            packed[j * stride + c] = 0;
    }
};
// Resolve neighbor width at launch; offsets are read once per query through
// their existing view. Both widths retain 64-bit byte-address arithmetic.
template <bool Wide> __device__ int packed_reference_index(IndexView indices, int64_t edge) {
    if constexpr (Wide)
        return int(static_cast<const int64_t *>(indices.data)[edge]);
    else
        return static_cast<const int32_t *>(indices.data)[edge];
}
template <int Stride>
__device__ void load_packed_reference(const float *packed, int j, float (&record)[Stride]) {
#pragma unroll
    for (int chunk = 0; chunk < Stride / 4; ++chunk) {
        const float4 value = reinterpret_cast<const float4 *>(packed)[j * (Stride / 4) + chunk];
        record[4 * chunk] = value.x;
        record[4 * chunk + 1] = value.y;
        record[4 * chunk + 2] = value.z;
        record[4 * chunk + 3] = value.w;
    }
}
} // namespace sph_native
