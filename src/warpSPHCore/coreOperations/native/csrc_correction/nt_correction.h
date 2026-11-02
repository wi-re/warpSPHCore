#pragma once
#include "crk/native/csrc_correction/nt_correction.h"

namespace sph_native::neighborhood {
// Only corrected operations carry these pointers. Ordinary operation/loader
// layouts and their tuned dispatch remain untouched.
template <class T> struct CorrectionFields {
    crk::Fields<T> crk;
    const T* reference_omega = nullptr;
    const T* renormalization = nullptr;
};
} // namespace sph_native::neighborhood
