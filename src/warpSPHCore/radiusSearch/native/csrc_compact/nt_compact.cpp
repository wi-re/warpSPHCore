#include "nt_compact_backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <new>

namespace sph_native::compact {
namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::invalid_argument(message);
}
void buffer(ConstBuffer value, int64_t needed, size_t item) {
    require(needed >= 0 && value.size >= needed && value.size >= 0 &&
                uint64_t(value.size) <= SIZE_MAX / item &&
                (!needed || (value.data && uintptr_t(value.data) % item == 0)),
            "Invalid or undersized compact-hash input buffer");
}
void domain(const Context &context, const Problem &problem, Grid &grid) {
    if (context.device == Device::CPU) {
        numeric(problem, [&](auto scalar, auto dimension) {
            using T = decltype(scalar);
            for (int axis = 0; axis < dimension; ++axis) {
                grid.real<T>().lower[axis] = static_cast<const T *>(problem.domain.min.data)[axis];
                grid.real<T>().upper[axis] = static_cast<const T *>(problem.domain.max.data)[axis];
                const auto flag = static_cast<const uint8_t *>(problem.domain.periodic.data)[axis];
                require(flag <= 1, "Invalid domain periodicity");
                grid.periodic[axis] = bool(flag);
            }
        });
    }
#ifdef SPH_COMPACT_HAS_GPU
    else
        gpu_domain(context, problem, grid);
#endif
    numeric(problem, [&](auto scalar, auto dimension) {
        using T = decltype(scalar);
        for (int axis = 0; axis < dimension; ++axis) {
            const T extent = T(grid.real<T>().upper[axis]) - T(grid.real<T>().lower[axis]);
            require(finite(grid.real<T>().lower[axis]) && finite(grid.real<T>().upper[axis]) && finite(extent) &&
                        extent > 0,
                    "Invalid domain extent");
        }
    });
}
Summary summary(const Context &context, const Problem &problem) {
    Summary result{};
    if (context.device == Device::CPU)
        result = cpu_summary(problem);
#ifdef SPH_COMPACT_HAS_GPU
    else
        result = gpu_summary(context, problem);
#endif
    return result;
}
template <class T> T radius_bound(const Problem &problem, const Summary &maxima) {
    if (!problem.query.count || !problem.reference.count) return T(0);
    const auto &values = maxima.real<T>();
    const T radius = pairwise_support(values.query_max, values.reference_max, int(problem.properties.support));
    require(finite(radius) && radius >= T(0), "Invalid or overflowing search radius");
    return radius;
}
} // namespace

void validate_geometry(const Context &context, const Problem &problem) {
    require(context.device_index >= 0 && context.cpu_threads >= 0, "Invalid execution context");
    bool available = context.device == Device::CPU;
#ifdef SPH_COMPACT_HAS_GPU
#if defined(SPH_NATIVE_HIP) || defined(USE_ROCM)
    available |= context.device == Device::HIP;
#else
    available |= context.device == Device::CUDA;
#endif
#endif
    require(available, "Device backend is not compiled into the compact-hash library");
    require(context.device != Device::CPU || !context.stream,
            "CPU context cannot carry a GPU stream");
    require(problem.precision == Precision::Float32 || problem.precision == Precision::Float64,
            "Unknown precision");
    require(problem.dimension >= 1 && problem.dimension <= 3, "Dimension must be 1, 2 or 3");
    require(problem.query.count >= 0 && problem.query.count <= INT64_MAX / problem.dimension &&
                problem.reference.count >= 0 && problem.reference.count <= INT32_MAX,
            "Invalid population sizes");
    require(int(problem.properties.support) >= 11 && int(problem.properties.support) <= 16,
            "Unknown or unsupported support policy");
    const size_t item = problem.precision == Precision::Float32 ? sizeof(float) : sizeof(double);
    buffer(problem.query.positions, problem.query.count * problem.dimension, item);
    buffer(problem.reference.positions, problem.reference.count * problem.dimension, item);
    buffer(problem.query.supports, problem.query.count, item);
    buffer(problem.reference.supports, problem.reference.count, item);
    buffer(problem.domain.min, problem.dimension, item);
    buffer(problem.domain.max, problem.dimension, item);
    buffer(problem.domain.periodic, problem.dimension, 1);
}

Data::~Data() {
    for (void *allocation : allocations) {
        if (context.device == Device::CPU)
            ::operator delete(allocation);
#ifdef SPH_COMPACT_HAS_GPU
        else
            gpu_free(context, allocation);
#endif
    }
}
void *Data::allocate(size_t bytes) {
    if (!bytes)
        return nullptr;
    void *allocation = nullptr;
    if (context.device == Device::CPU)
        allocation = ::operator new(bytes);
#ifdef SPH_COMPACT_HAS_GPU
    else
        allocation = gpu_allocate(context, bytes);
#endif
    try {
        allocations.push_back(allocation);
    } catch (...) {
        if (context.device == Device::CPU)
            ::operator delete(allocation);
#ifdef SPH_COMPACT_HAS_GPU
        else
            gpu_free(context, allocation);
#endif
        throw;
    }
    return allocation;
}
Data &Access::get(const CompactHash &index) {
    if (!index.data_)
        throw std::invalid_argument("Empty or moved-from compact hash");
    return *index.data_;
}
View view(const CompactHash &index) { return Access::get(index).view(); }

void validate_index(const Context &context, const Problem &problem, const Data &data) {
    validate_geometry(context, problem);
    require(context.device == data.context.device &&
                context.device_index == data.context.device_index &&
                problem.precision == data.precision && problem.dimension == data.dimension &&
                problem.reference.count == data.references,
            "Compact hash device, shape or precision mismatch");
}
void search(const Context &context, const Problem &problem, const CompactHash &index, Buffer output,
            bool collect) {
    const auto &data = Access::get(index);
    validate_index(context, problem, data);
    const auto written_bytes =
        size_t(collect ? output.size : problem.query.count) * (collect ? 8 : 4);
    auto overlaps = [&](const void *input, size_t bytes) {
        if (!written_bytes || !bytes)
            return false;
        const auto left = uintptr_t(output.data), right = uintptr_t(input);
        return left <= right ? right - left < written_bytes : left - right < bytes;
    };
    require(!overlaps(data.particles, size_t(data.references) * sizeof(ParticleKey)) &&
                !overlaps(data.cells, size_t(data.occupied) * sizeof(Cell)) &&
                !overlaps(data.buckets, size_t(data.grid.buckets) * sizeof(Bucket)),
            "Search output overlaps compact hash storage");

    if (context.device == Device::CPU) {
        cpu_search(context, problem, data, output, collect);
    }
#ifdef SPH_COMPACT_HAS_GPU
    else
        gpu_search(context, problem, data, output, collect);
#endif
}
} // namespace sph_native::compact

namespace sph_native {
CompactHash::CompactHash() = default;
CompactHash::~CompactHash() = default;
CompactHash::CompactHash(CompactHash &&) noexcept = default;
CompactHash &CompactHash::operator=(CompactHash &&) noexcept = default;
compact::View CompactHash::view() const { return compact::Access::get(*this).view(); }
SearchStatistics CompactHash::statistics() const {
    const auto &data = compact::Access::get(*this);
    return {data.references,   data.occupied,
            data.grid.buckets, {data.grid.cells[0], data.grid.cells[1], data.grid.cells[2]},
            data.precision == Precision::Float32 ? double(data.grid.real<float>().width) : data.grid.real<double>().width,
            data.precision == Precision::Float32 ? double(data.grid.real<float>().radius_bound) : data.grid.real<double>().radius_bound};
}
CompactHash build_compact_hash(const Context &context, const Problem &problem,
                               SearchOptions options) {
    using namespace compact;
    validate_geometry(context, problem);
    require(options.hash_buckets >= 0 &&
                uint64_t(options.hash_buckets) <= SIZE_MAX / sizeof(Bucket),
            "Invalid hash bucket count");
    CompactHash index;
    index.data_ = std::make_unique<Data>();
    auto &data = *index.data_;
    data.context = context;
    data.precision = problem.precision;
    data.dimension = problem.dimension;
    data.references = problem.reference.count;
    numeric(problem, [&](auto scalar, auto) {
        using T = decltype(scalar);
        new (&data.grid.real<T>()) GridReal<T>{};
    });
    domain(context, problem, data.grid);
    const auto maxima = summary(context, problem);
    numeric(problem, [&](auto scalar, auto) {
        using T = decltype(scalar);
        auto &real = data.grid.real<T>();
        T width = radius_bound<T>(problem, maxima), largest_extent = 0;
        for (int axis = 0; axis < problem.dimension; ++axis)
            largest_extent = std::max(largest_extent, real.upper[axis] - real.lower[axis]);
        constexpr T epsilon = std::numeric_limits<T>::epsilon();
        width = std::max({width, largest_extent / T(1048576), std::sqrt(std::numeric_limits<T>::min())});
        real.radius_bound = width;
        real.width = std::nextafter(width + T(128) * epsilon * largest_extent, std::numeric_limits<T>::max());
        require(finite(real.width) && real.width > T(0), "Invalid cell width");
        for (int axis = 0; axis < problem.dimension; ++axis) {
            const T ratio = (real.upper[axis] - real.lower[axis]) / real.width;
            data.grid.cells[axis] = std::max<int64_t>(1, int64_t(data.grid.periodic[axis] ? std::floor(ratio) : std::ceil(ratio)));
        }
    });
    data.grid.buckets =
        options.hash_buckets ? options.hash_buckets : std::max<int64_t>(1, data.references);
    require(uint64_t(data.references) <= SIZE_MAX / sizeof(ParticleKey) &&
                uint64_t(data.references) <= SIZE_MAX / sizeof(Cell),
            "Compact hash allocation overflow");
    data.particles =
        static_cast<ParticleKey *>(data.allocate(size_t(data.references) * sizeof(ParticleKey)));
    data.buckets = static_cast<Bucket *>(data.allocate(size_t(data.grid.buckets) * sizeof(Bucket)));
    if (context.device == Device::CPU)
        cpu_build(context, problem, data);
#ifdef SPH_COMPACT_HAS_GPU
    else
        gpu_build(context, problem, data);
#endif
    return index;
}
} // namespace sph_native
