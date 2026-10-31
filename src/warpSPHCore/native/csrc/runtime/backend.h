#pragma once
#include "api.h"
#include "dataTypes/state.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <stdexcept>
#include <limits>
namespace sph_native::runtime {
enum class Task { Forward, Adjoint, Count, Collect };
struct Prepared {
    std::array<ConstBuffer, OUTPUT> inputs{};
    std::array<int64_t, OUTPUT> sizes{};
    uint64_t required = 0;
    int64_t rows = 0, references = 0, output_size = 0;
    int dimension = 0, fields = 1, outputs = 1;
    int offsets_bits = 64, indices_bits = 64;
    Precision precision;
    Properties properties;
};
Prepared prepare(const Problem &, Task);
void check_buffer(ConstBuffer, int64_t, size_t, const char *);
void check_context(const Context &);
void zero(const Context &, void *, size_t);
void cpu_run(Task, const Context &, const Prepared &, Buffer, ConstBuffer, const Adjoints &);
#ifdef SPH_NATIVE_HAS_GPU
void gpu_run(Task, const Context &, const Prepared &, Buffer, ConstBuffer, const Adjoints &);
void gpu_zero(const Context &, void *, size_t);
#endif
inline bool floating(int slot) {
    return slot == QX || slot == RX || slot == QH || slot == RH || slot == MASS || slot == LO ||
           slot == HI || slot == QVALUE || slot == RVALUE || slot == DENSITY || slot == QDENSITY;
}
inline size_t element_size(const Prepared &p, int slot) {
    if (floating(slot))
        return p.precision == Precision::Float32 ? sizeof(float) : sizeof(double);
    if (slot == PERIODIC)
        return sizeof(bool);
    if (slot == OFFSETS)
        return p.offsets_bits / 8;
    if (slot == NEIGHBORS)
        return p.indices_bits / 8;
    return sizeof(int32_t);
}
inline uint64_t used_gradients(const Properties &p) {
    uint64_t mask = gradient_bit(QX) | gradient_bit(RX) | gradient_bit(QH) | gradient_bit(RH) |
                    gradient_bit(MASS) | gradient_bit(LO) | gradient_bit(HI);
    if (p.operation != Operation::Density)
        mask |= gradient_bit(DENSITY);
    if (p.operation != Operation::Density && p.operation != Operation::Covariance)
        mask |= gradient_bit(RVALUE);
    if (p.operation != Operation::Density && p.operation != Operation::Covariance &&
        p.operation != Operation::Interpolate) {
        if (p.scheme != Scheme::Naive || p.operation == Operation::Laplacian)
            mask |= gradient_bit(QVALUE);
        if (p.scheme == Scheme::Symmetric)
            mask |= gradient_bit(QDENSITY);
    }
    return mask;
}
template <class F> void numeric(const Prepared &p, const F &f) {
    if (p.precision == Precision::Float32) {
        if (p.dimension == 1)
            f(float{}, std::integral_constant<int, 1>{});
        else if (p.dimension == 2)
            f(float{}, std::integral_constant<int, 2>{});
        else
            f(float{}, std::integral_constant<int, 3>{});
    } else {
        if (p.dimension == 1)
            f(double{}, std::integral_constant<int, 1>{});
        else if (p.dimension == 2)
            f(double{}, std::integral_constant<int, 2>{});
        else
            f(double{}, std::integral_constant<int, 3>{});
    }
}
template <class T, int D> RowArgs<T, D> row_args(const Prepared &p, Buffer output) {
    RowArgs<T, D> a{};
    auto scalar = [&](int slot) { return static_cast<const T *>(p.inputs[slot].data); };
    a.qx = scalar(QX);
    a.rx = scalar(RX);
    a.qh = scalar(QH);
    a.rh = scalar(RH);
    a.mass = scalar(MASS);
    a.lo = scalar(LO);
    a.hi = scalar(HI);
    a.qvalue = scalar(QVALUE);
    a.rvalue = scalar(RVALUE);
    a.density = scalar(DENSITY);
    a.qdensity = scalar(QDENSITY);
    a.periodic = static_cast<const bool *>(p.inputs[PERIODIC].data);
    a.qkind = static_cast<const int32_t *>(p.inputs[QKIND].data);
    a.rkind = static_cast<const int32_t *>(p.inputs[RKIND].data);
    a.offsets = {p.inputs[OFFSETS].data, p.offsets_bits == 64};
    a.neighbors = {p.inputs[NEIGHBORS].data, p.indices_bits == 64};
    a.counts = const_cast<int32_t *>(static_cast<const int32_t *>(p.inputs[COUNTS].data));
    a.output = static_cast<T *>(output.data);
    a.references = p.references;
    a.fields = p.fields;
    a.outputs = p.outputs;
    a.operation = int(p.properties.operation);
    a.support = int(p.properties.support);
    a.direction = p.properties.direction;
    a.kernel = int(p.properties.kernel);
    a.scheme = int(p.properties.scheme);
    a.laplacian_mode = int(p.properties.laplacian);
    a.dot_mode = p.properties.divergence_dot;
    return a;
}
} // namespace sph_native::runtime
