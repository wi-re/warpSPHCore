#pragma once
#include "config.h"

namespace sph_native {
struct IndexView {
    const void *data = nullptr;
    bool wide = true;
    SPH_HD int64_t operator[](int64_t i) const {
        return wide ? static_cast<const int64_t *>(data)[i] : static_cast<const int32_t *>(data)[i];
    }
};

} // namespace sph_native
