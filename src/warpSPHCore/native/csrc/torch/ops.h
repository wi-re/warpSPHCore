#pragma once
#include "torch/types.h"
#include <array>
namespace sph_native {
// Call ATen schemas through the versioned stable C dispatcher. No ATen/c10
// C++ types or symbols cross the extension boundary.
template <class R, class... Args>
R call(const char *name, const char *overload, const Args &...args) {
    std::array<StableIValue, sizeof...(Args)> stack{stable::detail::from(args)...};
    const auto status = torch_call_dispatcher(name, overload, stack.data(), TORCH_ABI_VERSION);
    STD_TORCH_CHECK(status == 0, "Stable dispatcher failed: ", name, ".", overload);
    return stable::detail::to<R>(stack[0]);
}
inline bool all(const Tensor &x) {
    const auto result = call<Tensor>("aten::all", "", x);
    bool value;
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_item_bool(result.get(), &value));
    return value;
}
inline int64_t sum_int(const Tensor &x) {
    const auto result = stable::sum(x);
    int64_t value;
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(result.get(), &value));
    return value;
}
inline Tensor wide(const Tensor &x) { return stable::to(x, ScalarType::Long); }
inline Tensor exclusive_scan(const Tensor &x) {
    auto cumulative = call<Tensor>("aten::cumsum", "", x, int64_t(0), std::optional<ScalarType>{});
    if (x.numel() == 0)
        return cumulative;
    auto prefix = stable::narrow(cumulative, 0, 0, x.numel() - 1);
    return call<Tensor>("aten::cat", "", Tensors{stable::new_zeros(x, {1}), prefix}, int64_t(0));
}
inline Tensor ge_zero(const Tensor &x) {
    return call<Tensor>("aten::ge", "Tensor", x, stable::new_zeros(x, {}));
}
} // namespace sph_native
