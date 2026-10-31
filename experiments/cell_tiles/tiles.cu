// Experimental cell-to-cell interpolation. Numerical helpers are shared with
// production native code; sorting/planning and timing live in the Python driver.
#include "execution/cooperative.cuh"
#include "kernels/kernel.h"
#include "math/queryGeometry.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

struct TileCase {
    int dimension, precision, particles, cells, tasks, grid;
    void *stream;
    const void *positions, *supports, *masses, *densities, *values;
    const int64_t *cell_offsets;
    const int32_t *neighbors, *images, *prefix, *task_cells, *task_begin, *task_end;
    void *packed, *output;
};
namespace {
using namespace sph_native;
template <class T> constexpr T machine_epsilon = std::numeric_limits<T>::epsilon();
template <int D> __device__ int64_t morton(const int (&coordinate)[D]) {
    int64_t key = 0;
#pragma unroll
    for (int bit = 0; bit < 21; ++bit)
#pragma unroll
        for (int d = 0; d < D; ++d)
            key |= int64_t((coordinate[d] >> bit) & 1) << (D * bit + d);
    return key;
}
template <int D> __device__ void decode(int64_t key, int (&coordinate)[D]) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
        coordinate[d] = 0;
#pragma unroll
        for (int bit = 0; bit < 21; ++bit)
            coordinate[d] |= int((key >> (D * bit + d)) & 1) << bit;
    }
}
template <class T, int D>
__global__ void keys_kernel(const T *positions, int n, int grid, int64_t *keys) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int coordinate[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
        coordinate[d] = min(grid - 1, int(positions[i * D + d] * T(grid)));
    keys[i] = morton(coordinate);
}
template <int D> constexpr int stencil_size = D == 1 ? 3 : (D == 2 ? 9 : 27);
template <int D>
__global__ void neighbors_kernel(const int64_t *keys, int cells, int grid, int32_t *table,
                                 int32_t *images) {
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= cells * stencil_size<D>)
        return;
    const int cell = entry / stencil_size<D>;
    int offset = entry % stencil_size<D>, coordinate[D];
    decode<D>(keys[cell], coordinate);
#pragma unroll
    for (int d = 0; d < D; ++d) {
        const int raw = coordinate[d] + offset % 3 - 1;
        images[entry * D + d] = raw < 0 ? -1 : (raw >= grid ? 1 : 0);
        coordinate[d] = (raw + grid) % grid;
        offset /= 3;
    }
    const auto key = morton(coordinate);
    int low = 0, high = cells;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (keys[mid] < key)
            low = mid + 1;
        else
            high = mid;
    }
    table[entry] = low < cells && keys[low] == key ? low : -1;
}
// SoA records are compact and preserve coalescing across reference particles.
// Position plus weighted D-component field takes 2*D scalars per particle.
template <class T, int D> __global__ void prepare_kernel(TileCase a) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= a.particles)
        return;
    const auto *positions = static_cast<const T *>(a.positions);
    const auto *values = static_cast<const T *>(a.values);
    const T volume = static_cast<const T *>(a.masses)[j] / static_cast<const T *>(a.densities)[j];
    auto *packed = static_cast<T *>(a.packed);
#pragma unroll
    for (int d = 0; d < D; ++d) {
        packed[d * a.particles + j] = positions[j * D + d];
        packed[(D + d) * a.particles + j] = values[j * D + d] * volume;
    }
}
// One block owns a bounded chunk of one occupied Morton cell. Every reference
// tile is reused by all its query groups. Inactive query groups must still take
// part in both barriers; only active groups calculate or write an output.
template <class T, int D, int Block, int Width, bool Shared, bool CellImage, bool Squared = false>
__global__ void interpolate_kernel(TileCase a) {
    __shared__ T reference[2 * D][Block];
    const int task = blockIdx.x, lane = threadIdx.x % Width;
    const int query = a.task_begin[task] + threadIdx.x / Width;
    const bool active = query < a.task_end[task];
    const auto *positions = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    // Fill dimensions explicitly so the same source compiles in 1D/2D/3D.
    T lo[D], hi[D], point[D];
    bool periodic[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        lo[d] = 0;
        hi[d] = 1;
        periodic[d] = true;
        point[d] = active ? positions[query * D + d] : T(0);
    }
    const QueryGeometry<T, D> geometry(point, lo, hi, periodic);
    const Wendland2Primal<T, D> kernel(active ? static_cast<const T *>(a.supports)[query] : T(1));
    const T squared_cutoff =
        kernel.h * kernel.h * (T(1) + T(4) * machine_epsilon<T>);
    T sum[D] = {};
    const int cell = a.task_cells[task];
    for (int s = 0; s < stencil_size<D>; ++s) {
        const int neighbor = a.neighbors[cell * stencil_size<D> + s];
        if (neighbor < 0)
            continue;
        const int begin = int(a.cell_offsets[neighbor]), end = int(a.cell_offsets[neighbor + 1]);
        for (int base = begin; base < end; base += Block) {
            const int size = min(Block, end - base);
            if constexpr (Shared) {
                if (threadIdx.x < size) {
#pragma unroll
                    for (int c = 0; c < 2 * D; ++c)
                        reference[c][threadIdx.x] = packed[c * a.particles + base + threadIdx.x];
                }
                __syncthreads();
            }
            if (active) {
                for (int slot = lane; slot < size; slot += Width) {
                    T position[D], dx[D];
#pragma unroll
                    for (int d = 0; d < D; ++d)
                        position[d] =
                            Shared ? reference[d][slot] : packed[d * a.particles + base + slot];
                    T radius;
                    if constexpr (CellImage) {
                        // The planner selects the periodic image once per cell
                        // pair. Valid pairs need no floor/wrap work in this loop.
                        T squared = 0;
#pragma unroll
                        for (int d = 0; d < D; ++d) {
                            const T delta = (point[d] - position[d]) -
                                            T(a.images[(cell * stencil_size<D> + s) * D + d]);
                            squared += delta * delta;
                        }
                        if constexpr (Squared)
                            if (squared > squared_cutoff)
                                continue;
                        radius = scalar_sqrt(squared);
                        if (radius > kernel.h)
                            continue;
                    } else {
                        radius = geometry.distance(position, dx);
                    }
                    const T weight = kernel.value(radius);
#pragma unroll
                    for (int d = 0; d < D; ++d) {
                        const T value = Shared ? reference[D + d][slot]
                                               : packed[(D + d) * a.particles + base + slot];
                        sum[d] += weight * value;
                    }
                }
            }
            if constexpr (Shared)
                __syncthreads();
        }
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        const T value = group_sum<Width>(sum[d]);
        if (active && lane == 0)
            static_cast<T *>(a.output)[query * D + d] = value;
    }
}
// Flatten the cell stencil's small prefix array, not the particle CSR.
// Loading a reference tile may span several neighboring cells. This reduces
// synchronization from twice per neighboring cell to twice per full tile.
template <class T, int D, int Block, int Width, bool Squared = false>
__global__ void flat_interpolate_kernel(TileCase a) {
    __shared__ T source[3 * D][Block];
    const int task = blockIdx.x, lane = threadIdx.x % Width;
    const int query = a.task_begin[task] + threadIdx.x / Width;
    const bool active = query < a.task_end[task];
    const auto *positions = static_cast<const T *>(a.positions);
    const auto *packed = static_cast<const T *>(a.packed);
    T point[D], sum[D] = {};
#pragma unroll
    for (int d = 0; d < D; ++d)
        point[d] = active ? positions[query * D + d] : T(0);
    const Wendland2Primal<T, D> kernel(active ? static_cast<const T *>(a.supports)[query] : T(1));
    const T squared_cutoff =
        kernel.h * kernel.h * (T(1) + T(4) * machine_epsilon<T>);
    const int cell = a.task_cells[task];
    const int *prefix = a.prefix + cell * (stencil_size<D> + 1);
    const int total = prefix[stencil_size<D>];
    for (int base = 0; base < total; base += Block) {
        const int size = min(Block, total - base);
        if (threadIdx.x < size) {
            const int local = base + threadIdx.x;
            int low = 0, high = stencil_size<D>;
            while (low < high) {
                const int mid = (low + high) / 2;
                if (prefix[mid + 1] <= local)
                    low = mid + 1;
                else
                    high = mid;
            }
            const int neighbor = a.neighbors[cell * stencil_size<D> + low];
            const int j = int(a.cell_offsets[neighbor]) + local - prefix[low];
#pragma unroll
            for (int d = 0; d < D; ++d) {
                source[d][threadIdx.x] = packed[d * a.particles + j];
                source[D + d][threadIdx.x] = packed[(D + d) * a.particles + j];
                source[2 * D + d][threadIdx.x] =
                    T(a.images[(cell * stencil_size<D> + low) * D + d]);
            }
        }
        __syncthreads();
        if (active) {
            for (int slot = lane; slot < size; slot += Width) {
                T squared = 0;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    const T delta = (point[d] - source[d][slot]) - source[2 * D + d][slot];
                    squared += delta * delta;
                }
                if constexpr (Squared)
                    if (squared > squared_cutoff)
                        continue;
                const T radius = scalar_sqrt(squared);
                if (radius > kernel.h)
                    continue;
                const T weight = kernel.value(radius);
#pragma unroll
                for (int d = 0; d < D; ++d)
                    sum[d] += weight * source[D + d][slot];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
        const T value = group_sum<Width>(sum[d]);
        if (active && lane == 0)
            static_cast<T *>(a.output)[query * D + d] = value;
    }
}
template <class T, int D, int Block, int Width> void launch(const TileCase &a, int mode) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (mode == 6)
        interpolate_kernel<T, D, Block, Width, true, true, true><<<a.tasks, Block, 0, stream>>>(a);
    else if (mode == 7)
        flat_interpolate_kernel<T, D, Block, Width, true><<<a.tasks, Block, 0, stream>>>(a);
    else if (mode == 8)
        interpolate_kernel<T, D, Block, Width, false, true, true><<<a.tasks, Block, 0, stream>>>(a);
    else if (mode == 5)
        flat_interpolate_kernel<T, D, Block, Width><<<a.tasks, Block, 0, stream>>>(a);
    else if (mode == 1)
        interpolate_kernel<T, D, Block, Width, true, false><<<a.tasks, Block, 0, stream>>>(a);
    else if (mode == 0)
        interpolate_kernel<T, D, Block, Width, false, false><<<a.tasks, Block, 0, stream>>>(a);
    else if (mode == 3)
        interpolate_kernel<T, D, Block, Width, true, true><<<a.tasks, Block, 0, stream>>>(a);
    else
        interpolate_kernel<T, D, Block, Width, false, true><<<a.tasks, Block, 0, stream>>>(a);
}
template <class T, int D, int Block> void launch_width(const TileCase &a, int mode, int width) {
    switch (width) {
    case 1:
        launch<T, D, Block, 1>(a, mode);
        break;
    case 2:
        launch<T, D, Block, 2>(a, mode);
        break;
    case 4:
        launch<T, D, Block, 4>(a, mode);
        break;
    case 8:
        launch<T, D, Block, 8>(a, mode);
        break;
    }
}
template <class T, int D>
void dispatch(const TileCase &a, int mode, int block, int width, bool prepare) {
    const auto stream = static_cast<cudaStream_t>(a.stream);
    if (prepare)
        prepare_kernel<T, D><<<(a.particles + 255) / 256, 256, 0, stream>>>(a);
    if (mode == 2)
        return;
    switch (block) {
    case 32:
        launch_width<T, D, 32>(a, mode, width);
        break;
    case 64:
        launch_width<T, D, 64>(a, mode, width);
        break;
    case 128:
        launch_width<T, D, 128>(a, mode, width);
        break;
    case 256:
        launch_width<T, D, 256>(a, mode, width);
        break;
    }
}
} // namespace
#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT int tile_keys(const void *x, int n, int dimension, int precision, int grid, int64_t *keys,
                     void *raw_stream) {
    const auto stream = static_cast<cudaStream_t>(raw_stream);
#define KEYS(D, T)                                                                                 \
    keys_kernel<T, D><<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const T *>(x), n, grid, keys)
#define DIMENSION(T)                                                                               \
    switch (dimension) {                                                                           \
    case 1:                                                                                        \
        KEYS(1, T);                                                                                \
        break;                                                                                     \
    case 2:                                                                                        \
        KEYS(2, T);                                                                                \
        break;                                                                                     \
    case 3:                                                                                        \
        KEYS(3, T);                                                                                \
        break;                                                                                     \
    }
    if (precision == 0) {
        DIMENSION(float);
    } else {
        DIMENSION(double);
    }
#undef DIMENSION
#undef KEYS
    return int(cudaGetLastError());
}
EXPORT int tile_neighbors(const int64_t *keys, int cells, int dimension, int grid, int32_t *table,
                          int32_t *images, void *raw_stream) {
    const auto stream = static_cast<cudaStream_t>(raw_stream);
#define NEIGHBORS(D)                                                                               \
    neighbors_kernel<D><<<(cells * stencil_size<D> + 255) / 256, 256, 0, stream>>>(                \
        keys, cells, grid, table, images)
    switch (dimension) {
    case 1:
        NEIGHBORS(1);
        break;
    case 2:
        NEIGHBORS(2);
        break;
    case 3:
        NEIGHBORS(3);
        break;
    }
#undef NEIGHBORS
    return int(cudaGetLastError());
}
EXPORT int tile_interpolate(const TileCase *a, int mode, int block, int width, int prepare,
                            int iterations) {
#define RUN(D, T) dispatch<T, D>(*a, mode, block, width, prepare)
#define DIMENSION(T)                                                                               \
    switch (a->dimension) {                                                                        \
    case 1:                                                                                        \
        RUN(1, T);                                                                                 \
        break;                                                                                     \
    case 2:                                                                                        \
        RUN(2, T);                                                                                 \
        break;                                                                                     \
    case 3:                                                                                        \
        RUN(3, T);                                                                                 \
        break;                                                                                     \
    }
    for (int i = 0; i < iterations; ++i) {
        if (a->precision == 0) {
            DIMENSION(float);
        } else {
            DIMENSION(double);
        }
    }
#undef DIMENSION
#undef RUN
    return int(cudaGetLastError());
}
