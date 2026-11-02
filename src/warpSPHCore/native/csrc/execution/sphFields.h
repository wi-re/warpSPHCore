#pragma once

#include "execution/neighborhood.h"
#include "math/queryGeometry.h"
#include "kernels/value.h"
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

// Only general operations carry runtime kernel metadata. Keep it out of field
// views: reference loaders do not need it, and specialized policies should
// retain their original object layout. The empty base costs no storage.
template <class T, bool General> struct KernelMetadata {};
template <class T> struct KernelMetadata<T, true> {
    const T* reference_supports = nullptr;
    const T* query_density = nullptr;
    int kernel = 0, support = 11, scheme = 3;
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

// A CPU pair can read source fields in place. Keeping a pointer and one scalar
// avoids copying/padding a whole field record for each edge, particularly for
// matrix inputs. Packed GPU records retain their separate loader above.
template <class T, Weight Kind> struct ReferenceView {
    const T* data;
    const T* values;
    T scale;

    SPH_HD T weight() const { return Kind == Weight::WeightedField ? T(1) : scale; }
    SPH_HD T value(int c) const {
        if constexpr (Kind == Weight::WeightedField) return values[c] * scale;
        else return values[c];
    }
};

template <class T, int D, int Fields, Weight Kind> struct BorrowedReferenceLoader {
    SphFields<T, D, Fields> fields;

    SPH_HD bool allowed(int64_t j) const {
        return !fields.reference_kinds || reference_allowed(fields.reference_kinds[j], fields.direction);
    }

    SPH_HD ReferenceView<T, Kind> load(int64_t j) const {
        T scale = T(0);
        if constexpr (Kind == Weight::Mass) scale = fields.mass[j];
        else if constexpr (Kind == Weight::Volume || Kind == Weight::WeightedField)
            scale = fields.volumes ? fields.volumes[j] : fields.mass[j] / fields.density[j];
        const T* values = nullptr;
        if constexpr (Fields > 0) values = fields.reference_values + j * Fields;
        return {fields.reference_positions + j * D, values, scale};
    }
};

template <class T, int D, int Fields, bool General = false>
struct SphOperation : KernelMetadata<T, General> {
    static constexpr int field_count = Fields;
    static constexpr bool uses_query_values = false;
    SphFields<T, D, Fields> fields;
    using Query = PreparedQuery<T, D, Fields>;
    using Reference = ReferenceRecord<T, D, Fields>;

    template <class Args> void configure(const Args& a) {
        if constexpr (General) {
            this->reference_supports = a.rh;
            this->query_density = a.qdensity;
            this->kernel = a.kernel;
            this->support = a.support;
            this->scheme = a.scheme;
        }
    }

    // A neighborhood supplies the original directed reference ID even when
    // records have been packed or gathered through a different adjacency.
    // This lets general support rules use h_j without changing record layouts.
    template <class Record>
    SPH_HD void evaluate(const Query& q, const Record& r, Neighbor edge,
                         T (&dx)[D], T& value, T& gradient) const {
        const T distance = q.geometry.distance(r.data, dx);
        if constexpr (!General) {
            q.kernel.evaluate(distance, value, gradient);
        } else {
            // Non-Difference gradient schemes can still share a Gather
            // kernel across the row. Reuse its coefficients rather than
            // reconstructing them for every edge.
            if (this->kernel == 0 && this->support == 11) {
                q.kernel.evaluate(distance, value, gradient);
                return;
            }
            const T hj = this->reference_supports[edge.reference];
            if (this->support == 14 || this->support == 15) {
                T reference_value, reference_gradient;
                kernel_pair_value<T, D>(distance, q.support, this->kernel, value, gradient);
                kernel_pair_value<T, D>(distance, hj, this->kernel,
                                         reference_value, reference_gradient);
                value = (value + reference_value) / T(2);
                gradient = (gradient + reference_gradient) / T(2);
            } else {
                const T h = pairwise_support(q.support, hj, this->support);
                kernel_pair_value<T, D>(distance, h, this->kernel, value, gradient);
            }
        }
    }

    template <class Record>
    SPH_HD T value(const Query& q, const Record& r, Neighbor edge) const {
        if constexpr (!General) {
            T dx[D];
            return q.kernel.value(q.geometry.distance(r.data, dx));
        } else {
            T dx[D];
            const T distance = q.geometry.distance(r.data, dx);
            const T hj = this->reference_supports[edge.reference];
            if (this->support == 14 || this->support == 15)
                return (kernel_value<T, D>(distance, q.support, this->kernel) +
                        kernel_value<T, D>(distance, hj, this->kernel)) / T(2);
            return kernel_value<T, D>(distance,
                pairwise_support(q.support, hj, this->support), this->kernel);
        }
    }

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
