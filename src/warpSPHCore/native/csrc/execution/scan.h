#pragma once
#include "config.h"
#include <stdexcept>

namespace sph_native {
struct Context;
struct ConstBuffer;
struct Buffer;
namespace runtime {
int64_t gpu_scan(const Context &, int64_t, ConstBuffer, Buffer);
}
struct ScanSum {
    int64_t value;
    bool valid;
};
// Counts and offsets are nonnegative. Check before addition to avoid signed overflow
// on both the CPU and GPU; callers decide how to report an invalid scan.
SPH_HD constexpr ScanSum checked_scan_add(int64_t left, int64_t right) {
    if (left < 0 || right < 0 || left > INT64_MAX - right)
        return {0, false};
    return {left + right, true};
}
// Host scan over borrowed buffers. The public adapters validate capacities and
// non-overlap first. As on GPU, outputs are unspecified when validation fails.
inline int64_t exclusive_scan_cpu(const int32_t *counts, int64_t *offsets, int64_t rows) {
    int64_t total = 0;
    for (int64_t i = 0; i < rows; ++i) {
        const auto [sum, valid] = checked_scan_add(total, counts[i]);
        if (!valid)
            throw std::invalid_argument("Invalid counts or CSR size overflow");
        offsets[i] = total;
        total = sum;
    }
    return total;
}
} // namespace sph_native
