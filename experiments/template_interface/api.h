#pragma once
#include <cstdint>

// Private experiment ABI. No Torch headers or tensor-content checks are used
// here; the Python owner supplies valid, contiguous borrowed buffers.
struct Problem {
    int dimension, precision, work, queries, references, wide, direction;
    void* stream;
    const void *qx, *rx, *h, *mass, *density, *qvalue, *rvalue, *volume;
    const int32_t *qkind, *rkind;
    const void *lower, *upper;
    const bool* periodic;
};

struct Topology {
    // mode 0: explicit CSR, mode 1: per-query ranges, mode 2: shared query tiles.
    int mode, tasks, wide;
    const int64_t* offsets;
    const int32_t* counts;
    const void* indices;
    const int32_t *begins, *ends, *query_begin, *query_end;
};

struct Outputs {
    void *values, *cache, *packed;
    int32_t *counts, *winners;
};

struct Derivatives {
    const void* cotangent;
    // Tangents and gradients use the same seven independent tensor slots.
    // Missing tangents mean zero; missing gradient outputs mean unrequested.
    const void *tx, *trx, *th, *tm, *trho, *tqvalue, *trvalue;
    void *gx, *grx, *gh, *gm, *grho, *gqvalue, *grvalue;
    void* final_bars;
};
