#pragma once
#include <cstdint>
#include <memory>
#include "dataTypes/slots.h"
#include "dataTypes/compactHash.h"

#if defined(_WIN32)
#if defined(SPH_NATIVE_BUILDING_LIBRARY)
#define SPH_NATIVE_API __declspec(dllexport)
#else
#define SPH_NATIVE_API __declspec(dllimport)
#endif
#else
#define SPH_NATIVE_API __attribute__((visibility("default")))
#endif

namespace sph_native {
namespace compact {
struct Data;
struct Access;
} // namespace compact
enum class Precision { Float32, Float64 };
enum class Device { CPU, CUDA, HIP };
enum class Operation {
    Interpolate = 1,
    Gradient,
    Divergence,
    Curl,
    Laplacian,
    Density,
    Covariance
};
enum class Kernel {
    Wendland2 = 0,
    Wendland4 = 1,
    Wendland6 = 2,
    Poly6 = 20,
    Spiky = 21,
    CubicSpline = 30,
    QuarticSpline = 31,
    QuinticSpline = 32,
    B8 = 33,
    B7 = 34,
    HOCT4 = 35,
    Gaussian = 36,
    Viscosity = 40,
    Cohesion = 41,
    Adhesion = 42
};
enum class Support { Gather = 11, Scatter, MeanSymmetric, KernelMeanSymmetric, SuperSymmetric, PartialSymmetric };
enum class Scheme { Naive = 1, Symmetric, Difference, Summation };
enum class Laplacian { Naive = 1, Brookshaw, Dot, Default };

// Non-owning contiguous storage. Sizes are capacities in elements, not bytes.
// Every pointer must be accessible on the Context's device and remain alive
// until its stream has completed. Zero-sized views may have a null pointer.
struct ConstBuffer {
    const void *data = nullptr;
    int64_t size = 0;
};
struct Buffer {
    void *data = nullptr;
    int64_t size = 0;
};
struct Particles {
    int64_t count = 0;
    ConstBuffer positions, supports, masses, densities, kinds;
};
struct Domain {
    ConstBuffer min, max, periodic;
}; // periodic: D bool values
struct CSR {
    ConstBuffer offsets, counts, indices; // counts: int32, offsets: exclusive scan
    int offsets_bits = 64, indices_bits = 64;
};
struct Properties {
    Operation operation = Operation::Density;
    Kernel kernel = Kernel::Wendland2;
    Support support = Support::Gather;
    Scheme scheme = Scheme::Difference;
    Laplacian laplacian = Laplacian::Default;
    int direction = 0; // established OperationDirection IDs, 0 = TrueAllToToAll
    bool divergence_dot = false;
};
struct Problem {
    Precision precision = Precision::Float64;
    int dimension = 3, field_rank = 0; // scalar/vector/matrix: 0/1/2, each axis has extent D
    Particles query, reference;
    Domain domain;
    CSR adjacency;
    ConstBuffer query_values, reference_values;
    Properties properties;
};
struct Context {
    Device device = Device::CPU;
    int device_index = 0;
    void *stream = nullptr; // native CUDA/HIP stream; null means the default stream
    int cpu_threads = 0;    // OpenMP default, or explicit positive thread count
};
struct Adjoints {
    // Slots use dataTypes/slots.h. Only floating-point input slots are legal.
    // Exact aliases between gradient buffers are allowed and sum role contributions.
    Buffer slots[OUTPUT]{};
    uint64_t requested = 0;
    bool accumulate = false; // otherwise zero requested gradients before accumulating
};
constexpr uint64_t gradient_bit(Slot slot) { return uint64_t(1) << slot; }

struct SearchOptions {
    int64_t hash_buckets = 0; // zero chooses roughly one bucket per reference
};
struct SearchStatistics {
    int64_t references, occupied_cells, hash_buckets;
    int64_t grid_cells[3];
    double cell_width, radius_bound;
};
// Owns index memory on its build device. Particle arrays remain caller-owned.
// Construction completes on the supplied stream. Moving references across cells
// or growing the search radius beyond radius_bound requires rebuilding the index.
class SPH_NATIVE_API CompactHash {
  public:
    CompactHash();
    ~CompactHash();
    CompactHash(CompactHash &&) noexcept;
    CompactHash &operator=(CompactHash &&) noexcept;
    CompactHash(const CompactHash &) = delete;
    CompactHash &operator=(const CompactHash &) = delete;
    SearchStatistics statistics() const;
    // Borrowed read-only view for native frontend kernels. Retain the index
    // until work using its device pointers has completed on the caller's stream.
    compact::View view() const;

  private:
    std::unique_ptr<compact::Data> data_;
    friend struct compact::Access;
    friend CompactHash build_compact_hash(const Context &, const Problem &, SearchOptions);
};
SPH_NATIVE_API CompactHash build_compact_hash(const Context &, const Problem &, SearchOptions = {});
SPH_NATIVE_API void radius_count(const Context &, const Problem &, const CompactHash &,
                                 Buffer counts);
SPH_NATIVE_API void radius_collect(const Context &, const Problem &, const CompactHash &,
                                   Buffer indices);

SPH_NATIVE_API bool supports(Device device);
SPH_NATIVE_API int64_t output_elements(const Problem &problem);
SPH_NATIVE_API void forward(const Context &, const Problem &, Buffer output);
// Explicit VJP only: caller retains inputs and supplies cotangents and buffers.
SPH_NATIVE_API void adjoint(const Context &, const Problem &, ConstBuffer cotangent,
                            const Adjoints &);
// Compact-hash search: count -> scan/allocate -> collect. The overloads without
// an index build a temporary one; retain a CompactHash to reuse it. Scan returns
// the total to host; indexed count/collect and operator launches are stream-ordered.
SPH_NATIVE_API void radius_count(const Context &, const Problem &, Buffer counts);
// GPU scans keep prefixes on device; only the total and overflow status return to host.
SPH_NATIVE_API int64_t radius_scan(const Context &, int64_t rows, ConstBuffer counts,
                                   Buffer offsets);
SPH_NATIVE_API void radius_collect(const Context &, const Problem &, Buffer indices);
} // namespace sph_native
