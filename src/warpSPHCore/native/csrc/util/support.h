#pragma once
#include "config.h"
#include <cmath>

namespace sph_native {
// Warp float max treats a lone NaN as missing. CPU equality keeps the
// first signed zero; CUDA uses its native fmax intrinsic.
template<class T> SPH_HD T maximum_support(T a,T b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fmax(a,b);
#else
    return a>=b?a:(b==b?b:a);
#endif
}
// Taped adj_max sends finite ties and both-NaN inputs to the reference.
// Some legacy analytic products retain their established first-tie rule.
template<class T> SPH_HD bool maximum_support_query(T a,T b,bool first_tie=false) {
    return !std::isnan(a)&&(std::isnan(b)||(first_tie?a>=b:a>b));
}
template <typename T> SPH_HD T pairwise_support(T hi, T hj, int mode) {
    if (mode == 11)
        return hi;
    if (mode == 12)
        return hj;
    if (mode == 13)
        return (hi + hj) / T(2);
    return maximum_support(hi,hj);
}
} // namespace sph_native
