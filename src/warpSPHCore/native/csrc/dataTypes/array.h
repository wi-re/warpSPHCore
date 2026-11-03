#pragma once
#include "config.h"

namespace sph_native {
struct IndexView {
    const void *data = nullptr;
    bool wide = true;
    SPH_HD void set(int64_t i,int64_t value) const {
        if (wide) const_cast<int64_t*>(static_cast<const int64_t*>(data))[i]=value;
        else const_cast<int32_t*>(static_cast<const int32_t*>(data))[i]=int32_t(value);
    }
    SPH_HD int64_t operator[](int64_t i) const {
        return wide ? static_cast<const int64_t *>(data)[i] : static_cast<const int32_t *>(data)[i];
    }
};

} // namespace sph_native
