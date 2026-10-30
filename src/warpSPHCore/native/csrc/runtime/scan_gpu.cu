#include "runtime/backend.h"
#include "runtime/gpu_runtime.cuh"
#include "execution/scan.h"
#include <algorithm>
#include <vector>

namespace sph_native::runtime {
namespace {
constexpr int block_size = 256;
struct ScanResult {
    int64_t total;
    int invalid;
};

// Each block scans one tile. Padded lanes contribute zero, so its final lane
// also gives the tile sum for partial tiles. Grid-stride tiles support large
// buffers without assuming a particular CUDA or HIP grid-size limit.
template <class T>
__global__ void scan_tiles(const T *input, int64_t count, int64_t *output, int64_t *tile_sums,
                           ScanResult *result) {
    __shared__ int64_t values[block_size];
    const int lane = threadIdx.x;
    const int64_t tiles = (count - 1) / block_size + 1;
    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int64_t index = tile * block_size + lane;
        int64_t value = index < count ? int64_t(input[index]) : 0;
        if (value < 0) {
            atomicOr(&result->invalid, 1);
            value = 0;
        }
        values[lane] = value;
        __syncthreads();
        // All reads finish before any lane updates shared memory.
        for (int distance = 1; distance < block_size; distance *= 2) {
            const auto [sum, valid] =
                checked_scan_add(values[lane], lane >= distance ? values[lane - distance] : 0);
            __syncthreads();
            values[lane] = sum;
            if (!valid)
                atomicOr(&result->invalid, 2);
            __syncthreads();
        }
        if (index < count)
            output[index] = lane ? values[lane - 1] : 0;
        if (lane == block_size - 1) {
            if (tile_sums)
                tile_sums[tile] = values[lane];
            else
                result->total = values[lane];
        }
        __syncthreads();
    }
}

__global__ void add_tile_offsets(int64_t *output, int64_t count, const int64_t *tile_offsets,
                                 ScanResult *result) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += int64_t(blockDim.x) * gridDim.x) {
        const auto [sum, valid] = checked_scan_add(output[index], tile_offsets[index / block_size]);
        output[index] = sum;
        if (!valid)
            atomicOr(&result->invalid, 2);
    }
}

struct Allocation {
    void *data = nullptr;
    explicit Allocation(size_t bytes) { check(GPU(Malloc)(&data, bytes)); }
    ~Allocation() { (void)GPU(Free)(data); }
    Allocation(const Allocation &) = delete;
    Allocation &operator=(const Allocation &) = delete;
};

struct Level {
    int64_t count;
    int64_t *output;
};
} // namespace

int64_t gpu_scan(const Context &context, int64_t count, ConstBuffer counts, Buffer offsets) {
    if (!count)
        return 0;
    Guard guard(context.device_index);
    const auto stream = static_cast<NativeStream>(context.stream);
    // Workspace is geometric in the input size (two arrays per upper level).
    // The existing public buffer checks bound count*sizeof(int64_t), and this
    // geometric sum is smaller than count for the 256-element tile size.
    int64_t workspace_elements = 0;
    for (int64_t size = count; size > block_size;) {
        size = (size - 1) / block_size + 1;
        workspace_elements += 2 * size;
    }
    Allocation workspace(size_t(std::max<int64_t>(1, workspace_elements)) * sizeof(int64_t));
    Allocation status(sizeof(ScanResult));
    auto *result = static_cast<ScanResult *>(status.data);
    check(GPU(MemsetAsync)(result, 0, sizeof(ScanResult), stream));
    auto *cursor = static_cast<int64_t *>(workspace.data);
    std::vector<Level> levels;
    const int64_t *input = nullptr;
    int64_t size = count;
    auto *output = static_cast<int64_t *>(offsets.data);
    while (true) {
        const int64_t tiles = (size - 1) / block_size + 1;
        int64_t *sums = tiles > 1 ? cursor : nullptr;
        levels.push_back({size, output});
        const int grid = int(std::min<int64_t>(tiles, 65535));
        if (levels.size() == 1)
            scan_tiles<<<grid, block_size, 0, stream>>>(static_cast<const int32_t *>(counts.data),
                                                        size, output, sums, result);
        else
            scan_tiles<<<grid, block_size, 0, stream>>>(input, size, output, sums, result);
        check(GPU(GetLastError)());
        if (tiles == 1)
            break;
        input = sums;
        output = cursor + tiles;
        cursor += 2 * tiles;
        size = tiles;
    }
    // Propagate global offsets from the root scan down to the original rows.
    for (size_t upper = levels.size() - 1; upper > 0; --upper) {
        const auto [size, output] = levels[upper - 1];
        const int grid = int(std::min<int64_t>((size - 1) / block_size + 1, 65535));
        add_tile_offsets<<<grid, block_size, 0, stream>>>(output, size, levels[upper].output,
                                                          result);
        check(GPU(GetLastError)());
    }
    ScanResult host{};
    check(GPU(MemcpyAsync)(&host, result, sizeof(host), GPU(MemcpyDeviceToHost), stream));
    check(GPU(StreamSynchronize)(stream));
    if (host.invalid)
        throw std::invalid_argument("Invalid counts or CSR size overflow");
    return host.total;
}
} // namespace sph_native::runtime
