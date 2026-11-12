#pragma once
#include "config.h"
#include <cfloat>
namespace sph_native::compact {
template <class T> SPH_HD inline bool finite(T value) {
    constexpr T maximum = sizeof(T) == sizeof(float) ? T(FLT_MAX) : T(DBL_MAX);
    return value == value && value >= -maximum && value <= maximum;
}
} // namespace sph_native::compact
