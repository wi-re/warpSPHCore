#pragma once
#include "config.h"
namespace sph_native {
template <class T> SPH_HD void add_atomic(T *pointer, T value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicAdd(pointer, value);
#else
    T previous;
    __atomic_load(pointer, &previous, __ATOMIC_RELAXED);
    while (true) {
        T next = previous + value;
        if (__atomic_compare_exchange(pointer, &previous, &next, false, __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED))
            break;
    }
#endif
}
} // namespace sph_native
