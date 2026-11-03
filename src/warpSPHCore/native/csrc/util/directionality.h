#pragma once
#include "config.h"

namespace sph_native {
SPH_HD inline bool query_allowed(int32_t kind, int mode) {
    switch (mode) {
    case 0:
    case 9:
        return true; // mode 0 bypasses all direction checks in Warp
    case 1:
    case 3:
    case 6:
    case 11:
        return kind == 0;
    case 2:
    case 4:
    case 8:
    case 12:
        return kind == 1;
    case 5:
    case 7:
    case 10:
        return kind == 2;
    case 13:
    case 14:
        return kind != 2;
    default:
        return false;
    }
}
SPH_HD inline bool reference_allowed(int32_t kind, int mode) {
    switch (mode) {
    case 0:
        return true;
    case 9:
    case 10:
    case 11:
    case 12:
        return kind != 2;
    case 1:
    case 2:
    case 5:
    case 13:
        return kind == 0;
    case 3:
    case 4:
    case 7:
    case 14:
        return kind == 1;
    case 6:
    case 8:
        return kind == 2;
    default:
        return false;
    }
}

} // namespace sph_native
