#include "nt_compact_backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

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
                grid.lower[axis] = static_cast<const T *>(problem.domain.min.data)[axis];
                grid.upper[axis] = static_cast<const T *>(problem.domain.max.data)[axis];
                const auto flag = static_cast<const uint8_t *>(problem.domain.periodic.data)[axis];
                require(flag <= 1, "Invalid domain periodicity");
                grid.periodic[axis] = bool(flag);
            }
        });
    }
#ifdef SPH_COMPACT_HAS_GPU
    else
        gpu_domain(context, problem, grid.lower, grid.upper, grid.periodic);
#endif
    numeric(problem, [&](auto scalar, auto dimension) {
        using T = decltype(scalar);
        for (int axis = 0; axis < dimension; ++axis) {
            const T extent = T(grid.upper[axis]) - T(grid.lower[axis]);
            require(finite(grid.lower[axis]) && finite(grid.upper[axis]) && finite(extent) &&
                        extent > 0,
                    "Invalid domain extent");
        }
    });
}
Summary summary(const Context &context, const Problem &problem, const Grid &grid,
                const int64_t *keys = nullptr) {
    Summary result{};
    if (context.device == Device::CPU)
        result = cpu_summary(problem, grid, keys);
#ifdef SPH_COMPACT_HAS_GPU
    else
        result = gpu_summary(context, problem, grid, keys);
#endif
    require(!(result.invalid & 2), "Reference cells changed; rebuild the compact hash");
    require(!result.invalid, "Nonfinite or invalid scalar contents, or unrepresentable periodic "
                             "coordinates in compact hash");
    return result;
}
double radius_bound(const Problem &problem, Summary maxima) {
    if (!problem.query.count || !problem.reference.count)
        return 0;
    double radius = 0;
    numeric(problem, [&](auto scalar, auto) {
        using T = decltype(scalar);
        radius = pairwise_support(T(maxima.query_max), T(maxima.reference_max),
                                  int(problem.properties.support));
    });
    require(finite(radius) && radius >= 0, "Invalid or overflowing search radius");
    return radius;
}
} // namespace

void validate_geometry(const Context &context, const Problem &problem) {
    require(context.device_index >= 0 && context.cpu_threads >= 0 &&
                (context.validation == Validation::Contents ||
                 context.validation == Validation::Metadata),
            "Invalid execution context");
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
    require(int(problem.properties.support) >= 11 && int(problem.properties.support) <= 15,
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
    Grid current{};
    domain(context, problem, current);
    for (int axis = 0; axis < problem.dimension; ++axis)
        require(current.lower[axis] == data.grid.lower[axis] &&
                    current.upper[axis] == data.grid.upper[axis] &&
                    current.periodic[axis] == data.grid.periodic[axis],
                "Domain changed; rebuild the compact hash");
    const auto maxima = summary(context, problem, data.grid, data.reference_keys);
    require(radius_bound(problem, maxima) <= data.radius_bound,
            "Search support exceeds indexed radius bound; rebuild the compact hash");
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
                !overlaps(data.reference_keys, size_t(data.references) * sizeof(int64_t)) &&
                !overlaps(data.cells, size_t(data.occupied) * sizeof(Cell)) &&
                !overlaps(data.buckets, size_t(data.grid.buckets) * sizeof(Bucket)),
            "Search output overlaps compact hash storage");

    if (context.device == Device::CPU) {
        int32_t error = 0;
        cpu_search(context, problem, data, output, collect, &error);
        require(!error, "Radius counts no longer match the geometry; recount before collection");
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
            data.grid.width,   data.radius_bound};
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
    domain(context, problem, data.grid);
    const auto maxima = summary(context, problem, data.grid);
    double width = radius_bound(problem, maxima), largest_extent = 0;
    for (int axis = 0; axis < problem.dimension; ++axis)
        largest_extent = std::max(largest_extent, data.grid.upper[axis] - data.grid.lower[axis]);
    // Cap each conceptual axis at 2^20 cells: 3D keys then fit int64 without
    // allocating a dense grid. Tiny supports remain correct with larger cells.
    const double epsilon = problem.precision == Precision::Float32 ? FLT_EPSILON : DBL_EPSILON;
    // Squared distances can underflow in the established scalar predicate.
    // Coarsen tiny grids so indexing cannot discard those accepted pairs.
    const double minimum_normal = problem.precision == Precision::Float32 ? FLT_MIN : DBL_MIN;
    width = std::max({width, largest_extent / 1048576.0, std::sqrt(minimum_normal)});
    data.radius_bound = width;
    // Keep this rounding reserve separate from the reusable support bound;
    // callers cannot consume it by gradually increasing their supports.
    data.grid.width = std::nextafter(width + 128 * epsilon * largest_extent, DBL_MAX);
    require(finite(data.grid.width) && data.grid.width > 0, "Invalid cell width");
    for (int axis = 0; axis < problem.dimension; ++axis) {
        const double ratio = (data.grid.upper[axis] - data.grid.lower[axis]) / data.grid.width;
        data.grid.cells[axis] = std::max<int64_t>(
            1, int64_t(data.grid.periodic[axis] ? std::floor(ratio) : std::ceil(ratio)));
    }
    data.grid.buckets =
        options.hash_buckets ? options.hash_buckets : std::max<int64_t>(1, data.references);
    require(uint64_t(data.references) <= SIZE_MAX / sizeof(ParticleKey) &&
                uint64_t(data.references) <= SIZE_MAX / sizeof(Cell),
            "Compact hash allocation overflow");
    data.particles =
        static_cast<ParticleKey *>(data.allocate(size_t(data.references) * sizeof(ParticleKey)));
    data.reference_keys =
        static_cast<int64_t *>(data.allocate(size_t(data.references) * sizeof(int64_t)));
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
