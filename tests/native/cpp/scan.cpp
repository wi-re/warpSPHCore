#include "scan_cases.h"
#include <iostream>

int main() {
    using namespace scan_checks;
    try {
        arithmetic();
        Context context{};
        for (const auto size : sizes) {
            const auto input = counts(size);
            std::vector<int64_t> output(size + 3, canary);
            const auto total =
                radius_scan(context, size, {input.data(), size}, {output.data(), size + 3});
            compare(input, output, total);
        }
        std::vector<int32_t> input{1, -1};
        std::vector<int64_t> output(2);
        rejects([&] { radius_scan(context, 2, {input.data(), 2}, {output.data(), 2}); });
        input[1] = 1;
        rejects([&] { radius_scan(context, 2, {input.data(), 1}, {output.data(), 2}); });
        rejects([&] { radius_scan(context, 2, {input.data(), 2}, {output.data(), 1}); });
        rejects([&] { radius_scan(context, 2, {output.data(), 2}, {output.data(), 2}); });
        rejects([&] { radius_scan(context, -1, {}, {}); });
        std::cout << "CPU CSR scan: boundaries, int64 offsets and invalid inputs passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
