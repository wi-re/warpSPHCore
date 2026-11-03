#pragma once
#include <cmath>
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPH_HD __host__ __device__
#else
#define SPH_HD
#endif
