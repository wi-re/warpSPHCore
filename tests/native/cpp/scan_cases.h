#pragma once
#include "api.h"
#include "execution/scan.h"
#include <limits>
#include <stdexcept>
#include <vector>

namespace scan_checks {
using namespace sph_native;
inline void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline std::vector<int32_t> counts(int64_t size) {
    std::vector<int32_t> result(size);
    for (int64_t i = 0; i < size; ++i)
        result[i] = i % 7 == 0 ? std::numeric_limits<int32_t>::max() : int32_t(i % 13);
    return result;
}
// Three extra elements check that the scan respects the logical row count,
// even when the caller supplies a larger output allocation.
constexpr int64_t canary = -123456789;
inline void compare(const std::vector<int32_t> &input, const std::vector<int64_t> &output,
                    int64_t total) {
    int64_t expected = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        require(output[i] == expected, "Incorrect exclusive scan offset");
        expected += input[i];
    }
    require(total == expected, "Incorrect scan total");
    for (size_t i = input.size(); i < output.size(); ++i)
        require(output[i] == canary, "Scan wrote beyond the logical output");
}
// Tile boundaries, partial tiles and a hierarchy requiring three scan levels.
inline constexpr int64_t sizes[] = {0, 1, 255, 256, 257, 65536, 65537};
inline void arithmetic() {
    constexpr auto maximum = std::numeric_limits<int64_t>::max();
    const auto [value, valid] = checked_scan_add(maximum, 0);
    require(valid && value == maximum, "Scan must retain int64 range");
    require(!checked_scan_add(maximum, 1).valid, "Scan must reject int64 overflow");
    require(!checked_scan_add(0, -1).valid, "Scan must reject negative counts");
}
template <class F> void rejects(F call) {
    try {
        call();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("Invalid scan arguments were accepted");
}
} // namespace scan_checks
