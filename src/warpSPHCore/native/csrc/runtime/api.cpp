#include "runtime/backend.h"
#include "kernels/radial.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <string>
namespace sph_native::runtime {
static void require(bool condition, const char *message) {
    if (!condition)
        throw std::invalid_argument(message);
}
static int64_t multiply(int64_t a, int64_t b) {
    require(a >= 0 && b >= 0 && (!b || a <= std::numeric_limits<int64_t>::max() / b),
            "Element count overflow");
    return a * b;
}
static size_t bytes(int64_t n, size_t item) {
    require(n >= 0 && uint64_t(n) <= std::numeric_limits<size_t>::max() / item,
            "Byte count overflow");
    return size_t(n) * item;
}
void check_buffer(ConstBuffer b, int64_t needed, size_t alignment, const char *name) {
    if (b.size < needed || b.size < 0 || (needed && !b.data) ||
        (needed && reinterpret_cast<uintptr_t>(b.data) % alignment))
        throw std::invalid_argument(std::string("Invalid or undersized ") + name);
    bytes(b.size, alignment);
}
static Prepared describe(const Problem &v, Task task) {
    Prepared p{};
    p.precision = v.precision;
    p.dimension = v.dimension;
    p.properties = v.properties;
    p.rows = v.query.count;
    p.references = v.reference.count;
    require(v.precision == Precision::Float32 || v.precision == Precision::Float64,
            "Unknown precision");
    require(v.dimension >= 1 && v.dimension <= 3, "Dimension must be 1, 2 or 3");
    require(v.field_rank >= 0 && v.field_rank <= 2, "Field rank must be 0, 1 or 2");
    require(p.rows >= 0 && p.references >= 0 && p.references <= std::numeric_limits<int32_t>::max(),
            "Invalid population sizes");
    auto op = v.properties.operation;
    int support = int(v.properties.support);
    require(support >= 11 && support <= 15, "Unknown or unsupported support policy");
    p.fields = 1;
    for (int i = 0; i < v.field_rank; ++i)
        p.fields *= v.dimension;
    if (task == Task::Forward || task == Task::Adjoint) {
        require(int(op) >= 1 && int(op) <= 7, "Unknown operation");
        require(valid_kernel(int(v.properties.kernel)), "Unknown kernel");
        require(v.properties.direction >= 0 && v.properties.direction <= 14,
                "Unknown operation direction");
        require(int(v.properties.scheme) >= 1 && int(v.properties.scheme) <= 4,
                "Unknown gradient scheme");
        require(int(v.properties.laplacian) >= 1 && int(v.properties.laplacian) <= 4,
                "Unknown Laplacian scheme");
        p.outputs = p.fields;
        if (op == Operation::Density)
            p.outputs = 1;
        if (op == Operation::Covariance)
            p.outputs = v.dimension * v.dimension;
        if (op == Operation::Gradient)
            p.outputs *= v.dimension;
        if (op == Operation::Divergence) {
            require(v.field_rank > 0, "Divergence requires a vector or matrix field");
            p.outputs /= v.dimension;
        }
        if (op == Operation::Curl) {
            require(v.dimension == 1 || v.field_rank > 0, "Curl requires a vector or matrix field");
            p.outputs =
                v.dimension == 1 ? 1 : (v.dimension == 2 ? std::max(1, p.fields / 2) : p.fields);
        }
        if (op == Operation::Laplacian && v.properties.laplacian == Laplacian::Dot)
            require(v.dimension == 1 || v.field_rank > 0,
                    "Dot Laplacian requires a vector or matrix field");
    }
    p.output_size = multiply(p.rows, p.outputs);
    p.sizes[QX] = multiply(p.rows, v.dimension);
    p.sizes[RX] = multiply(p.references, v.dimension);
    p.sizes[QH] = p.sizes[QKIND] = p.sizes[QDENSITY] = p.sizes[COUNTS] = p.sizes[OFFSETS] = p.rows;
    p.sizes[RH] = p.sizes[RKIND] = p.sizes[MASS] = p.sizes[DENSITY] = p.references;
    p.sizes[LO] = p.sizes[HI] = p.sizes[PERIODIC] = v.dimension;
    p.sizes[QVALUE] = multiply(p.rows, p.fields);
    p.sizes[RVALUE] = multiply(p.references, p.fields);
    return p;
}
Prepared prepare(const Problem &v, Task task) {
    auto p = describe(v, task);
    p.inputs = {v.query.positions,     v.reference.positions, v.query.supports,
                v.reference.supports,  v.reference.masses,    v.query.kinds,
                v.reference.kinds,     v.domain.min,          v.domain.max,
                v.domain.periodic,     v.adjacency.offsets,   v.adjacency.counts,
                v.adjacency.indices,   v.query_values,        v.reference_values,
                v.reference.densities, v.query.densities};
    p.offsets_bits = v.adjacency.offsets_bits;
    p.indices_bits = v.adjacency.indices_bits;
    require((p.offsets_bits == 32 || p.offsets_bits == 64) &&
                (p.indices_bits == 32 || p.indices_bits == 64),
            "CSR widths must be 32 or 64");
    p.required = gradient_bit(QX) | gradient_bit(RX) | gradient_bit(QH) | gradient_bit(RH) |
                 gradient_bit(LO) | gradient_bit(HI) | gradient_bit(PERIODIC);
    if (task == Task::Forward || task == Task::Adjoint) {
        p.required |= used_gradients(p.properties) | gradient_bit(QKIND) | gradient_bit(RKIND) |
                      gradient_bit(OFFSETS) | gradient_bit(COUNTS) | gradient_bit(NEIGHBORS);
        if (p.properties.operation != Operation::Interpolate &&
            p.properties.operation != Operation::Density &&
            p.properties.operation != Operation::Covariance)
            p.required |= gradient_bit(QVALUE);
    }
    if (task == Task::Collect)
        p.required |= gradient_bit(OFFSETS) | gradient_bit(COUNTS);
    for (int slot = 0; slot < OUTPUT; ++slot)
        if (p.required & (uint64_t(1) << slot))
            check_buffer(p.inputs[slot], p.sizes[slot], element_size(p, slot), "input buffer");
    // Indices are checked against the live count after reading the CSR offsets.
    return p;
}
void check_context(const Context &c) {
    require(c.cpu_threads >= 0 && c.device_index >= 0, "Invalid execution context");
    require(c.validation == Validation::Contents || c.validation == Validation::Metadata,
            "Unknown validation mode");
    require(supports(c.device), "Device backend is not compiled into this standalone library");
    if (c.device == Device::CPU)
        require(c.stream == nullptr, "CPU context cannot carry a GPU stream");
}
void read(const Context &c, void *dest, const void *src, size_t n) {
    if (!n)
        return;
    if (c.device == Device::CPU) {
        std::memcpy(dest, src, n);
        return;
    }
#ifdef SPH_NATIVE_HAS_GPU
    gpu_read(c, dest, src, n);
#endif
}
void write(const Context &c, void *dest, const void *src, size_t n) {
    if (!n)
        return;
    if (c.device == Device::CPU) {
        std::memcpy(dest, src, n);
        return;
    }
#ifdef SPH_NATIVE_HAS_GPU
    gpu_write(c, dest, src, n);
#endif
}
void zero(const Context &c, void *dest, size_t n) {
    if (!n)
        return;
    if (c.device == Device::CPU) {
        std::memset(dest, 0, n);
        return;
    }
#ifdef SPH_NATIVE_HAS_GPU
    gpu_zero(c, dest, n);
#endif
}
template <class T>
static void values(const Context &c, ConstBuffer input, int64_t n, bool positive,
                   bool nonnegative) {
    std::vector<T> host(size_t(n), T{});
    read(c, host.data(), input.data, bytes(n, sizeof(T)));
    for (T x : host)
        require(std::isfinite(x) && (!positive || x > T(0)) && (!nonnegative || x >= T(0)),
                "Nonfinite or invalid scalar contents");
}
void validate_contents(const Context &c, const Prepared &p, Task task) {
    if (c.validation == Validation::Contents) {
        for (int slot = 0; slot < OUTPUT; ++slot)
            if (floating(slot) && (p.required & (uint64_t(1) << slot))) {
                bool positive = slot == DENSITY || slot == QDENSITY,
                     nonnegative = slot == QH || slot == RH;
                if (p.precision == Precision::Float32)
                    values<float>(c, p.inputs[slot], p.sizes[slot], positive, nonnegative);
                else
                    values<double>(c, p.inputs[slot], p.sizes[slot], positive, nonnegative);
            }
        numeric(p, [&](auto value, auto dim) {
            using T = decltype(value);
            constexpr int D = decltype(dim)::value;
            T lo[D], hi[D];
            uint8_t periodic[D];
            read(c, lo, p.inputs[LO].data, sizeof(lo));
            read(c, hi, p.inputs[HI].data, sizeof(hi));
            static_assert(sizeof(bool) == 1);
            read(c, periodic, p.inputs[PERIODIC].data, sizeof(periodic));
            for (int d = 0; d < D; ++d)
                require(hi[d] > lo[d] && periodic[d] <= 1, "Invalid domain extent or periodicity");
        });
    }
    if (task == Task::Count)
        return;
    if (c.validation == Validation::Metadata && task != Task::Collect)
        return;
    std::vector<int32_t> counts(size_t(p.rows), 0);
    read(c, counts.data(), p.inputs[COUNTS].data, bytes(p.rows, 4));
    std::vector<int64_t> offsets(size_t(p.rows), 0);
    if (p.offsets_bits == 64)
        read(c, offsets.data(), p.inputs[OFFSETS].data, bytes(p.rows, 8));
    else {
        std::vector<int32_t> narrow(size_t(p.rows), 0);
        read(c, narrow.data(), p.inputs[OFFSETS].data, bytes(p.rows, 4));
        std::copy(narrow.begin(), narrow.end(), offsets.begin());
    }
    int64_t total = 0;
    for (int64_t i = 0; i < p.rows; ++i) {
        require(counts[i] >= 0 && offsets[i] == total, "Invalid CSR counts or exclusive offsets");
        require(total <= std::numeric_limits<int64_t>::max() - counts[i], "CSR size overflow");
        total += counts[i];
    }
    require(total <= p.inputs[NEIGHBORS].size, "CSR capacity does not cover live entries");
    if (task == Task::Collect)
        return;
    check_buffer(p.inputs[NEIGHBORS], total, p.indices_bits / 8, "CSR indices");
    if (p.indices_bits == 64) {
        std::vector<int64_t> ids(size_t(total), 0);
        read(c, ids.data(), p.inputs[NEIGHBORS].data, bytes(total, 8));
        for (auto j : ids)
            require(j >= 0 && j < p.references, "Invalid live CSR reference index");
    } else {
        std::vector<int32_t> ids(size_t(total), 0);
        read(c, ids.data(), p.inputs[NEIGHBORS].data, bytes(total, 4));
        for (auto j : ids)
            require(j >= 0 && j < p.references, "Invalid live CSR reference index");
    }
}
static bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
    if (!an || !bn)
        return false;
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
static void no_input_alias(const Prepared &p, Buffer out, size_t n) {
    for (int s = 0; s < OUTPUT; ++s)
        if (p.required & (uint64_t(1) << s)) {
            int64_t count = s == NEIGHBORS ? p.inputs[s].size : p.sizes[s];
            require(!overlaps(out.data, n, p.inputs[s].data, bytes(count, element_size(p, s))),
                    "Output must not overlap input storage");
        }
}
static void run(Task task, const Context &c, const Prepared &p, Buffer out, ConstBuffer cot = {},
                const Adjoints &bars = {}) {
    if (c.device == Device::CPU)
        cpu_run(task, c, p, out, cot, bars);
#ifdef SPH_NATIVE_HAS_GPU
    else
        gpu_run(task, c, p, out, cot, bars);
#endif
}
} // namespace sph_native::runtime

namespace sph_native {
using namespace runtime;
bool supports(Device device) {
    if (device == Device::CPU)
        return true;
#ifdef SPH_NATIVE_CUDA
    if (device == Device::CUDA)
        return true;
#endif
#ifdef SPH_NATIVE_HIP
    if (device == Device::HIP)
        return true;
#endif
    return false;
}
int64_t output_elements(const Problem &p) { return describe(p, Task::Forward).output_size; }
void forward(const Context &c, const Problem &view, Buffer out) {
    check_context(c);
    auto p = prepare(view, Task::Forward);
    size_t item = element_size(p, QX);
    check_buffer({out.data, out.size}, p.output_size, item, "output");
    no_input_alias(p, out, bytes(p.output_size, item));
    validate_contents(c, p, Task::Forward);
    run(Task::Forward, c, p, out);
}
void adjoint(const Context &c, const Problem &view, ConstBuffer cot, const Adjoints &bars) {
    check_context(c);
    auto p = prepare(view, Task::Adjoint);
    int k = int(p.properties.kernel);
    require(k != 21 && k != 35 && k != 40 && k != 42,
            "Cusp/singular kernel reverse mode is not qualified");
    size_t item = element_size(p, QX);
    check_buffer(cot, p.output_size, item, "cotangent");
    require((bars.requested >> OUTPUT) == 0, "Unknown gradient-mask bit");
    for (int s = 0; s < OUTPUT; ++s)
        if (bars.requested & (uint64_t(1) << s)) {
            require(floating(s), "Gradient mask may only request floating-point inputs");
            auto b = bars.slots[s];
            size_t n = bytes(p.sizes[s], item);
            check_buffer({b.data, b.size}, p.sizes[s], item, "gradient buffer");
            no_input_alias(p, b, n);
            require(!overlaps(b.data, n, cot.data, bytes(p.output_size, item)),
                    "Gradients must not overlap the cotangent");
            for (int t = 0; t < s; ++t)
                if (bars.requested & (uint64_t(1) << t)) {
                    auto previous = bars.slots[t];
                    size_t pn = bytes(p.sizes[t], item);
                    require(!overlaps(b.data, n, previous.data, pn) ||
                                (b.data == previous.data && n == pn),
                            "Gradient buffers may alias exactly, but not partially");
                }
        }
    validate_contents(c, p, Task::Adjoint);
    if (!bars.accumulate)
        for (int s = 0; s < OUTPUT; ++s)
            if (bars.requested & (uint64_t(1) << s))
                zero(c, bars.slots[s].data, bytes(p.sizes[s], item));
    run(Task::Adjoint, c, p, {}, cot, bars);
}
void radius_count(const Context &c, const Problem &view, Buffer out) {
    check_context(c);
    auto p = prepare(view, Task::Count);
    check_buffer({out.data, out.size}, p.rows, 4, "counts output");
    no_input_alias(p, out, bytes(p.rows, 4));
    validate_contents(c, p, Task::Count);
    run(Task::Count, c, p, out);
}
int64_t radius_scan(const Context &c, int64_t n, ConstBuffer counts, Buffer offsets) {
    check_context(c);
    require(n >= 0, "Invalid row count");
    check_buffer(counts, n, 4, "counts");
    check_buffer({offsets.data, offsets.size}, n, 8, "offsets");
    require(!overlaps(counts.data, bytes(n, 4), offsets.data, bytes(n, 8)),
            "Scan buffers must not overlap");
    std::vector<int32_t> host(size_t(n), 0);
    read(c, host.data(), counts.data, bytes(n, 4));
    std::vector<int64_t> prefix(size_t(n), 0);
    int64_t total = 0;
    for (int64_t i = 0; i < n; ++i) {
        require(host[i] >= 0 && total <= std::numeric_limits<int64_t>::max() - host[i],
                "Invalid counts or CSR size overflow");
        prefix[i] = total;
        total += host[i];
    }
    write(c, offsets.data, prefix.data(), bytes(n, 8));
    return total;
}
void radius_collect(const Context &c, const Problem &view, Buffer out) {
    check_context(c);
    auto p = prepare(view, Task::Collect);
    check_buffer({out.data, out.size}, 0, 8, "indices output");
    p.inputs[NEIGHBORS] = {out.data, out.size};
    p.required |= gradient_bit(NEIGHBORS);
    p.indices_bits = 64;
    // Output capacity is validated before any writes, even in Metadata mode.
    validate_contents(c, p, Task::Collect);
    auto inputs = p;
    inputs.required &= ~gradient_bit(NEIGHBORS);
    no_input_alias(inputs, out, bytes(out.size, 8));
    // Counts may sum to zero while output is null. The checked row never writes then.
    check_buffer({out.data, out.size}, out.size, 8, "indices output");
    run(Task::Collect, c, p, out);
}
} // namespace sph_native
