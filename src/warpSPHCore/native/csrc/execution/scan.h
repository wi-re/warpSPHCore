#pragma once
#include "config.h"

namespace sph_native {
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
} // namespace sph_native
