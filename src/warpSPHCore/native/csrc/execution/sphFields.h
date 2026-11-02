#pragma once

#include "execution/neighborhood.h"
#include "math/queryGeometry.h"
#include "kernels/kernel.h"
#include "util/directionality.h"
#include <type_traits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include "execution/device_runtime.cuh"
#endif

namespace sph_native::neighborhood {

enum class Weight { None, Mass, Volume, WeightedField };

// Borrowed tensor views. Absence is represented by nullptr, not dummy tensors.
// Position/support/domain are required; fields and optional outputs are chosen
// by the operation. Tensor ownership remains with the caller/AD context.
template <class T, int D, int Fields> struct SphFields {
    const T *query_positions, *reference_positions, *supports;
    const T *mass = nullptr, *density = nullptr;
    const T *query_values = nullptr, *reference_values = nullptr;
    const T *volumes = nullptr;
    const int32_t *query_kinds = nullptr, *reference_kinds = nullptr;
    const T *lower, *upper;
    const bool* periodic;
    int direction = 0;
    T* output = nullptr;
    T* cache = nullptr;
    int32_t* counts = nullptr;
    int32_t* winners = nullptr;
};

template <class T, int D, int Fields> struct PreparedQuery {
    int64_t index;
    QueryGeometry<T, D> geometry;
    Wendland2Primal<T, D> kernel;
    T support;
    T values[Fields ? Fields : 1] = {};
};

template <class T, int D, int Fields> struct ReferenceRecord {
    static constexpr int pitch = ((D + 1 + Fields + 3) / 4) * 4;
    T data[pitch];
    SPH_HD T weight() const { return data[D]; }
    SPH_HD T value(int component) const { return data[D + 1 + component]; }
};

// Plain interpolation consumes V_j*f_j rather than V_j and f_j separately.
// In 3D scalar FP32 this changes the aligned reference record from 32 to 16
// bytes. The operation can use the same weight()/value() interface either way.
template <class T, int D, int Fields> struct WeightedReferenceRecord {
    static constexpr int pitch = ((D + Fields + 3) / 4) * 4;
    T data[pitch];
    SPH_HD T weight() const { return T(1); }
    SPH_HD T value(int component) const { return data[D + component]; }
};

template <class T, int D, int Fields, Weight Kind, bool Packed = false>
struct ReferenceLoader {
    SphFields<T, D, Fields> fields;
    const T* packed = nullptr;
    using Record = std::conditional_t<Kind == Weight::WeightedField,
        WeightedReferenceRecord<T,D,Fields>,ReferenceRecord<T,D,Fields>>;

    SPH_HD bool allowed(int64_t j) const {
        return !fields.reference_kinds || reference_allowed(fields.reference_kinds[j], fields.direction);
    }

    SPH_HD Record load(int64_t j) const {
        Record result{};
        if constexpr (Packed) {
            const auto* source = packed + j * Record::pitch;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            if constexpr (sizeof(T) == 4) {
                for (int c = 0; c < Record::pitch; c += 4) {
                    const auto value = reinterpret_cast<const float4*>(source)[c / 4];
                    result.data[c] = value.x; result.data[c + 1] = value.y;
                    result.data[c + 2] = value.z; result.data[c + 3] = value.w;
                }
            } else {
                for (int c = 0; c < Record::pitch; c += 2) {
                    const auto value = reinterpret_cast<const double2*>(source)[c / 2];
                    result.data[c] = value.x; result.data[c + 1] = value.y;
                }
            }
#else
            for (int c = 0; c < Record::pitch; ++c) result.data[c] = source[c];
#endif
        } else {
            for (int d = 0; d < D; ++d)
                result.data[d] = fields.reference_positions[j * D + d];
            if constexpr (Kind == Weight::Mass)
                result.data[D] = fields.mass[j];
            else if constexpr (Kind == Weight::Volume)
                result.data[D] = fields.volumes ? fields.volumes[j] : fields.mass[j] / fields.density[j];
            if constexpr (Kind == Weight::WeightedField) {
                const T volume = fields.volumes ? fields.volumes[j] : fields.mass[j] / fields.density[j];
                for (int c = 0; c < Fields; ++c)
                    result.data[D + c] = fields.reference_values[j * Fields + c] * volume;
            } else {
                for (int c = 0; c < Fields; ++c)
                    result.data[D + 1 + c] = fields.reference_values[j * Fields + c];
            }
        }
        return result;
    }
};

template <class T, int D, int Fields> struct SphOperation {
    SphFields<T, D, Fields> fields;
    using Query = PreparedQuery<T, D, Fields>;
    using Reference = ReferenceRecord<T, D, Fields>;

    SPH_HD Query prepare(int64_t i) const {
        const T h = fields.supports[i];
        Query query{i, QueryGeometry<T, D>(fields.query_positions + i * D,
            fields.lower, fields.upper, fields.periodic), Wendland2Primal<T, D>(h), h};
        if (fields.query_values)
            for (int c = 0; c < Fields; ++c)
                query.values[c] = fields.query_values[i * Fields + c];
        return query;
    }

    SPH_HD bool allowed(int64_t i) const {
        return !fields.query_kinds || query_allowed(fields.query_kinds[i], fields.direction);
    }
};

} // namespace sph_native::neighborhood
